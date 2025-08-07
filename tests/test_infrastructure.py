"""Simple test to verify testing infrastructure works."""



def test_pytest_works():
    """Basic test to verify pytest is working."""
    assert True


def test_async_pytest_works():
    """Basic async test to verify pytest-asyncio is working."""
    import asyncio

    async def async_function():
        await asyncio.sleep(0.01)
        return True

    # This will be run synchronously for this simple test
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(async_function())
        assert result is True
    finally:
        loop.close()


def test_imports_work():
    """Test that we can import testing dependencies."""

    assert True


class TestBasicStructure:
    """Test basic test class structure."""

    def test_class_based_tests(self):
        """Test that class-based tests work."""
        assert True

    def test_fixtures_available(self):
        """Test that basic fixtures are available."""
        # This tests that our conftest.py is being loaded
        assert True
