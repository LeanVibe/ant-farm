# Dependency Injection Container for clean architecture

import asyncio
import inspect
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, get_type_hints, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

T = TypeVar('T')


class DependencyInjectionError(Exception):
    """Exception raised for dependency injection errors."""
    pass


class ServiceLifetime:
    """Service lifetime enumeration."""
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    SCOPED = "scoped"


@dataclass
class ServiceDescriptor:
    """Service registration descriptor."""
    service_type: Any
    implementation_type: Optional[Any]
    lifetime: str
    instance: Optional[Any] = None
    factory: Optional[Callable] = None


class DependencyContainer:
    """Lightweight dependency injection container."""

    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._scoped_services: Dict[str, Dict[Type, Any]] = {}
        self._current_scope: Optional[str] = None

    def register_singleton(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None,
                          instance: Optional[T] = None, factory: Optional[Callable[[], T]] = None) -> None:
        """Register a singleton service."""
        if implementation_type is None and instance is None and factory is None:
            implementation_type = service_type

        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            lifetime=ServiceLifetime.SINGLETON,
            instance=instance,
            factory=factory
        )

    def register_transient(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None,
                          factory: Optional[Callable[[], T]] = None) -> None:
        """Register a transient service."""
        if implementation_type is None and factory is None:
            implementation_type = service_type

        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            lifetime=ServiceLifetime.TRANSIENT,
            factory=factory
        )

    def register_scoped(self, service_type: Type[T], implementation_type: Optional[Type[T]] = None,
                       factory: Optional[Callable[[], T]] = None) -> None:
        """Register a scoped service."""
        if implementation_type is None and factory is None:
            implementation_type = service_type

        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            implementation_type=implementation_type,
            lifetime=ServiceLifetime.SCOPED,
            factory=factory
        )

    async def resolve(self, service_type: Type[T], scope_id: Optional[str] = None) -> Any:
        """Resolve a service instance."""
        if service_type not in self._services:
            raise DependencyInjectionError(f"Service {service_type} is not registered")

        descriptor = self._services[service_type]

        # Handle different lifetimes
        if descriptor.lifetime == ServiceLifetime.SINGLETON:
            if descriptor.instance is None:
                descriptor.instance = await self._create_instance(descriptor, scope_id)
            return descriptor.instance

        elif descriptor.lifetime == ServiceLifetime.TRANSIENT:
            instance = await self._create_instance(descriptor, scope_id)
            return instance

        elif descriptor.lifetime == ServiceLifetime.SCOPED:
            if scope_id is None:
                scope_id = self._current_scope or "default"

            if scope_id not in self._scoped_services:
                self._scoped_services[scope_id] = {}

            if service_type not in self._scoped_services[scope_id]:
                self._scoped_services[scope_id][service_type] = await self._create_instance(descriptor, scope_id)

            return self._scoped_services[scope_id][service_type]

        else:
            raise DependencyInjectionError(f"Unknown service lifetime: {descriptor.lifetime}")

    async def _create_instance(self, descriptor: ServiceDescriptor, scope_id: Optional[str] = None) -> Any:
        """Create a new instance of a service."""
        if descriptor.factory:
            # Use factory function
            if inspect.iscoroutinefunction(descriptor.factory):
                instance = await descriptor.factory()
            else:
                instance = descriptor.factory()
            if instance is None:
                raise DependencyInjectionError(f"Factory for {descriptor.service_type} returned None")
            return instance

        elif descriptor.instance:
            # Use provided instance
            return descriptor.instance

        else:
            # Create instance using constructor injection
            if descriptor.implementation_type is None:
                raise DependencyInjectionError(f"No implementation type for {descriptor.service_type}")
            return await self._create_with_constructor_injection(descriptor.implementation_type, scope_id)

    async def _create_with_constructor_injection(self, implementation_type: Any, scope_id: Optional[str] = None) -> Any:
        """Create instance with constructor dependency injection."""
        try:
            # Get constructor parameters
            init_signature = inspect.signature(implementation_type.__init__)
            parameters = init_signature.parameters

            # Skip 'self' parameter
            param_names = [name for name in parameters.keys() if name != 'self']
            param_types = get_type_hints(implementation_type.__init__)

            # Resolve dependencies
            kwargs = {}
            for param_name in param_names:
                if param_name in param_types:
                    param_type = param_types[param_name]
                    # Handle Optional types
                    if hasattr(param_type, '__origin__') and param_type.__origin__ is Union:
                        # Get the non-None type from Optional[T]
                        param_type = param_type.__args__[0] if param_type.__args__ else param_type

                    try:
                        kwargs[param_name] = await self.resolve(param_type, scope_id)
                    except DependencyInjectionError:
                        # If dependency cannot be resolved, try to get default value
                        if param_name in parameters and parameters[param_name].default != inspect.Parameter.empty:
                            kwargs[param_name] = parameters[param_name].default
                        else:
                            raise

            return implementation_type(**kwargs)

        except Exception as e:
            raise DependencyInjectionError(f"Failed to create instance of {implementation_type}: {str(e)}")

    @asynccontextmanager
    async def scope(self, scope_id: str):
        """Create a scoped context for scoped services."""
        old_scope = self._current_scope
        self._current_scope = scope_id

        try:
            yield
        finally:
            # Clean up scoped services
            if scope_id in self._scoped_services:
                del self._scoped_services[scope_id]
            self._current_scope = old_scope

    def clear_scope(self, scope_id: str) -> None:
        """Clear a specific scope."""
        if scope_id in self._scoped_services:
            del self._scoped_services[scope_id]

    def get_registered_services(self) -> List[Type]:
        """Get list of registered service types."""
        return list(self._services.keys())

    def is_registered(self, service_type: Type) -> bool:
        """Check if a service type is registered."""
        return service_type in self._services


# Global container instance
_container = DependencyContainer()


def get_container() -> DependencyContainer:
    """Get the global dependency injection container."""
    return _container


async def resolve(service_type: Type[T]) -> T:
    """Resolve a service from the global container."""
    return await _container.resolve(service_type)


def register_singleton(service_type: Type[T], implementation_type: Optional[Type[T]] = None,
                      instance: Optional[T] = None, factory: Optional[Callable[[], T]] = None) -> None:
    """Register a singleton service in the global container."""
    _container.register_singleton(service_type, implementation_type, instance, factory)


def register_transient(service_type: Type[T], implementation_type: Optional[Type[T]] = None,
                      factory: Optional[Callable[[], T]] = None) -> None:
    """Register a transient service in the global container."""
    _container.register_transient(service_type, implementation_type, factory)


def register_scoped(service_type: Type[T], implementation_type: Optional[Type[T]] = None,
                   factory: Optional[Callable[[], T]] = None) -> None:
    """Register a scoped service in the global container."""
    _container.register_scoped(service_type, implementation_type, factory)


# Convenience functions for common registrations
def register_core_services():
    """Register all core services with the container."""
    try:
        # Import concrete implementations
        from .orchestrator_refactored import AgentOrchestratorRefactored
        from .agent_registry import AgentRegistry
        from .agent_spawner import AgentSpawner
        from .health_monitor import HealthMonitor
        from ..agents.agent_factory import AgentFactory

        # Register services with appropriate lifetimes
        register_singleton(IAgentOrchestrator, AgentOrchestratorRefactored)
        register_singleton(IAgentFactory, AgentFactory)
        register_singleton(IHealthMonitor, HealthMonitor)

        # Register transient services
        register_transient(IAgentRegistry, AgentRegistry)
        register_transient(IAgentSpawner, AgentSpawner)

    except ImportError as e:
        # Handle missing implementations gracefully
        print(f"Warning: Some services could not be registered: {e}")
        pass


async def initialize_container():
    """Initialize the dependency injection container."""
    register_core_services()
    # TODO: Initialize services when implementations are ready
    return None


async def shutdown_container():
    """Shutdown the dependency injection container."""
    try:
        # TODO: Stop services gracefully when implementations are ready
        pass
    except Exception as e:
        print(f"Error during container shutdown: {e}")


# Context manager for container lifecycle
@asynccontextmanager
async def container_lifecycle():
    """Context manager for container initialization and shutdown."""
    try:
        yield await initialize_container()
    finally:
        await shutdown_container()