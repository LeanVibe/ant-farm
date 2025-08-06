// Service Worker for PWA functionality
const CACHE_NAME = 'hive-dashboard-v1';
const STATIC_CACHE_URLS = [
    '/',
    '/index.html',
    '/styles/dashboard.css',
    '/components/hive-dashboard.js',
    '/components/agent-status.js',
    '/components/task-board.js',
    '/components/message-flow.js',
    '/components/system-metrics.js',
    '/components/context-explorer.js',
    '/components/log-viewer.js',
    '/services/websocket-service.js',
    '/services/api-service.js',
    '/manifest.json'
];

const API_CACHE_URLS = [
    '/api/v1/status',
    '/api/v1/health',
    '/api/v1/agents',
    '/api/v1/tasks',
    '/api/v1/metrics'
];

// Install event
self.addEventListener('install', (event) => {
    console.log('Service Worker installing...');
    
    event.waitUntil(
        Promise.all([
            caches.open(CACHE_NAME).then((cache) => {
                console.log('Caching static files...');
                return cache.addAll(STATIC_CACHE_URLS);
            }),
            self.skipWaiting()
        ])
    );
});

// Activate event
self.addEventListener('activate', (event) => {
    console.log('Service Worker activating...');
    
    event.waitUntil(
        Promise.all([
            // Clean up old caches
            caches.keys().then((cacheNames) => {
                return Promise.all(
                    cacheNames
                        .filter((cacheName) => cacheName !== CACHE_NAME)
                        .map((cacheName) => caches.delete(cacheName))
                );
            }),
            self.clients.claim()
        ])
    );
});

// Fetch event
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);
    
    // Handle different types of requests
    if (request.method === 'GET') {
        if (url.pathname.startsWith('/api/')) {
            // API requests - cache with network-first strategy
            event.respondWith(handleApiRequest(request));
        } else if (STATIC_CACHE_URLS.includes(url.pathname)) {
            // Static files - cache-first strategy
            event.respondWith(handleStaticRequest(request));
        } else {
            // Other requests - network-first with fallback
            event.respondWith(handleGenericRequest(request));
        }
    }
});

// Handle API requests with network-first strategy
async function handleApiRequest(request) {
    const cache = await caches.open(CACHE_NAME);
    
    try {
        // Try network first
        const networkResponse = await fetch(request);
        
        // Cache successful responses
        if (networkResponse.ok) {
            cache.put(request, networkResponse.clone());
        }
        
        return networkResponse;
    } catch (error) {
        console.log('Network failed, trying cache...', error);
        
        // Fall back to cache
        const cachedResponse = await cache.match(request);
        if (cachedResponse) {
            return cachedResponse;
        }
        
        // Return offline page for HTML requests
        if (request.headers.get('accept')?.includes('text/html')) {
            return new Response(
                generateOfflinePage(),
                { 
                    headers: { 'Content-Type': 'text/html' },
                    status: 200
                }
            );
        }
        
        // Return error response for API requests
        return new Response(
            JSON.stringify({ 
                error: 'Offline', 
                message: 'API not available offline' 
            }),
            { 
                headers: { 'Content-Type': 'application/json' },
                status: 503
            }
        );
    }
}

// Handle static files with cache-first strategy
async function handleStaticRequest(request) {
    const cache = await caches.open(CACHE_NAME);
    const cachedResponse = await cache.match(request);
    
    if (cachedResponse) {
        return cachedResponse;
    }
    
    try {
        const networkResponse = await fetch(request);
        cache.put(request, networkResponse.clone());
        return networkResponse;
    } catch (error) {
        console.log('Failed to fetch static resource:', error);
        throw error;
    }
}

// Handle generic requests
async function handleGenericRequest(request) {
    try {
        return await fetch(request);
    } catch (error) {
        const cache = await caches.open(CACHE_NAME);
        const cachedResponse = await cache.match(request);
        
        if (cachedResponse) {
            return cachedResponse;
        }
        
        // Return offline page for navigation requests
        if (request.mode === 'navigate') {
            return new Response(
                generateOfflinePage(),
                { 
                    headers: { 'Content-Type': 'text/html' },
                    status: 200
                }
            );
        }
        
        throw error;
    }
}

// Generate offline page HTML
function generateOfflinePage() {
    return `
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>LeanVibe Hive - Offline</title>
            <style>
                body {
                    font-family: 'Segoe UI', sans-serif;
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                    color: #ffffff;
                    margin: 0;
                    padding: 0;
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    text-align: center;
                }
                .offline-container {
                    max-width: 500px;
                    padding: 2rem;
                }
                .offline-icon {
                    font-size: 4rem;
                    margin-bottom: 1rem;
                }
                .offline-title {
                    font-size: 2rem;
                    margin-bottom: 1rem;
                    color: #00d4aa;
                }
                .offline-message {
                    font-size: 1.1rem;
                    margin-bottom: 2rem;
                    opacity: 0.8;
                }
                .retry-button {
                    background: #00d4aa;
                    color: #1a1a2e;
                    border: none;
                    padding: 1rem 2rem;
                    border-radius: 6px;
                    font-size: 1rem;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s ease;
                }
                .retry-button:hover {
                    background: #00e5bb;
                    transform: translateY(-2px);
                }
                .features-list {
                    text-align: left;
                    margin-top: 2rem;
                    padding: 1rem;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 8px;
                }
                .features-list h3 {
                    margin-top: 0;
                    color: #00d4aa;
                }
                .features-list ul {
                    margin: 0;
                    padding-left: 1.5rem;
                }
                .features-list li {
                    margin-bottom: 0.5rem;
                    opacity: 0.8;
                }
            </style>
        </head>
        <body>
            <div class="offline-container">
                <div class="offline-icon">📡</div>
                <h1 class="offline-title">You're Offline</h1>
                <p class="offline-message">
                    The LeanVibe Agent Hive dashboard is not available right now. 
                    Check your internet connection and try again.
                </p>
                <button class="retry-button" onclick="window.location.reload()">
                    Try Again
                </button>
                
                <div class="features-list">
                    <h3>Available Offline:</h3>
                    <ul>
                        <li>Cached dashboard interface</li>
                        <li>Previous session data</li>
                        <li>Static documentation</li>
                        <li>PWA functionality</li>
                    </ul>
                </div>
            </div>
            
            <script>
                // Auto-retry when connection is restored
                window.addEventListener('online', () => {
                    console.log('Connection restored');
                    window.location.reload();
                });
                
                // Update UI based on connection status
                function updateConnectionStatus() {
                    if (navigator.onLine) {
                        document.querySelector('.offline-message').textContent = 
                            'Connection restored! Refreshing...';
                        setTimeout(() => window.location.reload(), 1000);
                    }
                }
                
                // Check connection status periodically
                setInterval(updateConnectionStatus, 5000);
            </script>
        </body>
        </html>
    `;
}

// Handle background sync
self.addEventListener('sync', (event) => {
    if (event.tag === 'background-sync') {
        event.waitUntil(doBackgroundSync());
    }
});

async function doBackgroundSync() {
    console.log('Performing background sync...');
    
    try {
        // Sync pending data when connection is restored
        const cache = await caches.open(CACHE_NAME);
        
        // Update critical data
        const criticalUrls = [
            '/api/v1/status',
            '/api/v1/agents',
            '/api/v1/tasks'
        ];
        
        for (const url of criticalUrls) {
            try {
                const response = await fetch(url);
                if (response.ok) {
                    await cache.put(url, response.clone());
                }
            } catch (error) {
                console.log(`Failed to sync ${url}:`, error);
            }
        }
        
        console.log('Background sync completed');
    } catch (error) {
        console.error('Background sync failed:', error);
    }
}

// Handle push notifications
self.addEventListener('push', (event) => {
    if (!event.data) return;
    
    const data = event.data.json();
    const options = {
        body: data.body || 'New notification from LeanVibe Hive',
        icon: '/icons/icon-192x192.png',
        badge: '/icons/badge-72x72.png',
        tag: data.tag || 'hive-notification',
        data: data.data || {},
        actions: [
            {
                action: 'open',
                title: 'Open Dashboard'
            },
            {
                action: 'dismiss',
                title: 'Dismiss'
            }
        ],
        vibrate: [200, 100, 200],
        renotify: true,
        requireInteraction: data.requireInteraction || false
    };
    
    event.waitUntil(
        self.registration.showNotification(data.title || 'LeanVibe Hive', options)
    );
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
    event.notification.close();
    
    if (event.action === 'open' || !event.action) {
        event.waitUntil(
            clients.matchAll({ type: 'window' }).then((clientList) => {
                // If dashboard is already open, focus it
                for (const client of clientList) {
                    if (client.url.includes('/') && 'focus' in client) {
                        return client.focus();
                    }
                }
                
                // Otherwise, open new window
                if (clients.openWindow) {
                    return clients.openWindow('/');
                }
            })
        );
    }
});

// Handle messages from the main thread
self.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }
});

console.log('Service Worker loaded');