// PWA Installation and Management Utility
class PWAManager {
    constructor() {
        this.deferredPrompt = null;
        this.isInstalled = false;
        this.updateAvailable = false;
        this.registration = null;
        
        this.init();
    }

    async init() {
        // Register service worker
        await this.registerServiceWorker();
        
        // Set up PWA install prompt
        this.setupInstallPrompt();
        
        // Set up update handling
        this.setupUpdateHandling();
        
        // Check if already installed
        this.checkInstallationStatus();
        
        // Set up periodic sync
        this.setupPeriodicSync();
    }

    async registerServiceWorker() {
        if ('serviceWorker' in navigator) {
            try {
                this.registration = await navigator.serviceWorker.register('/sw.js', {
                    scope: '/'
                });
                
                console.log('[PWA] Service Worker registered:', this.registration);
                
                // Listen for updates
                this.registration.addEventListener('updatefound', () => {
                    this.handleServiceWorkerUpdate();
                });
                
                // Handle messages from service worker
                navigator.serviceWorker.addEventListener('message', (event) => {
                    this.handleServiceWorkerMessage(event);
                });
                
            } catch (error) {
                console.error('[PWA] Service Worker registration failed:', error);
            }
        } else {
            console.warn('[PWA] Service Workers not supported');
        }
    }

    setupInstallPrompt() {
        // Listen for beforeinstallprompt event
        window.addEventListener('beforeinstallprompt', (event) => {
            console.log('[PWA] Install prompt available');
            
            // Prevent the mini-infobar from appearing
            event.preventDefault();
            
            // Store the event for later use
            this.deferredPrompt = event;
            
            // Show custom install button
            this.showInstallButton();
        });

        // Listen for app installed event
        window.addEventListener('appinstalled', () => {
            console.log('[PWA] App installed');
            this.isInstalled = true;
            this.hideInstallButton();
            this.showInstallSuccessMessage();
        });
    }

    setupUpdateHandling() {
        // Check for updates when page loads
        if (this.registration) {
            this.registration.update();
        }

        // Listen for controlling service worker changes
        navigator.serviceWorker.addEventListener('controllerchange', () => {
            console.log('[PWA] New service worker took control');
            this.showUpdateSuccessMessage();
        });
    }

    handleServiceWorkerUpdate() {
        const newWorker = this.registration.installing;
        if (!newWorker) return;

        newWorker.addEventListener('statechange', () => {
            if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                console.log('[PWA] Update available');
                this.updateAvailable = true;
                this.showUpdateNotification();
            }
        });
    }

    handleServiceWorkerMessage(event) {
        const { data } = event;
        
        if (data.type === 'CACHE_UPDATED') {
            console.log('[PWA] Cache updated');
        }
        
        if (data.type === 'OFFLINE_READY') {
            console.log('[PWA] App ready for offline use');
            this.showOfflineReadyMessage();
        }
    }

    checkInstallationStatus() {
        // Check if running in standalone mode
        if (window.matchMedia('(display-mode: standalone)').matches) {
            this.isInstalled = true;
            console.log('[PWA] App is installed and running standalone');
        }
        
        // Check if running as PWA on iOS
        if (window.navigator.standalone === true) {
            this.isInstalled = true;
            console.log('[PWA] App is installed on iOS');
        }
    }

    async setupPeriodicSync() {
        if ('serviceWorker' in navigator && 'sync' in window.ServiceWorkerRegistration.prototype) {
            try {
                await this.registration.sync.register('background-sync');
                console.log('[PWA] Background sync registered');
            } catch (error) {
                console.warn('[PWA] Background sync not supported:', error);
            }
        }
    }

    // UI Methods
    showInstallButton() {
        let installButton = document.getElementById('pwa-install-button');
        
        if (!installButton) {
            installButton = this.createInstallButton();
            document.body.appendChild(installButton);
        }
        
        installButton.style.display = 'block';
        
        // Auto-hide after 10 seconds
        setTimeout(() => {
            if (installButton && !this.isInstalled) {
                installButton.style.opacity = '0.7';
            }
        }, 10000);
    }

    createInstallButton() {
        const button = document.createElement('button');
        button.id = 'pwa-install-button';
        button.innerHTML = `
            <span>📱</span>
            <span>Install App</span>
        `;
        
        button.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: linear-gradient(135deg, #00d4aa, #00e5bb);
            color: #1a1a2e;
            border: none;
            border-radius: 50px;
            padding: 12px 20px;
            font-weight: 600;
            font-size: 14px;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(0, 212, 170, 0.3);
            transition: all 0.3s ease;
            z-index: 1000;
            display: none;
            align-items: center;
            gap: 8px;
        `;
        
        button.addEventListener('mouseover', () => {
            button.style.transform = 'translateY(-2px)';
            button.style.boxShadow = '0 6px 25px rgba(0, 212, 170, 0.4)';
        });
        
        button.addEventListener('mouseout', () => {
            button.style.transform = 'translateY(0)';
            button.style.boxShadow = '0 4px 20px rgba(0, 212, 170, 0.3)';
        });
        
        button.addEventListener('click', () => {
            this.promptInstall();
        });
        
        return button;
    }

    hideInstallButton() {
        const installButton = document.getElementById('pwa-install-button');
        if (installButton) {
            installButton.style.display = 'none';
        }
    }

    async promptInstall() {
        if (!this.deferredPrompt) {
            this.showManualInstallInstructions();
            return;
        }

        try {
            // Show the install prompt
            this.deferredPrompt.prompt();
            
            // Wait for the user's response
            const { outcome } = await this.deferredPrompt.userChoice;
            
            console.log('[PWA] Install prompt outcome:', outcome);
            
            if (outcome === 'accepted') {
                console.log('[PWA] User accepted install');
            } else {
                console.log('[PWA] User dismissed install');
                this.showInstallDeclinedMessage();
            }
            
            // Clear the deferredPrompt
            this.deferredPrompt = null;
            this.hideInstallButton();
            
        } catch (error) {
            console.error('[PWA] Install prompt failed:', error);
            this.showManualInstallInstructions();
        }
    }

    showUpdateNotification() {
        const notification = this.createNotification(
            'Update Available',
            'A new version of the app is available. Update now?',
            [
                {
                    text: 'Update',
                    action: () => this.applyUpdate(),
                    primary: true
                },
                {
                    text: 'Later',
                    action: () => this.dismissNotification(),
                    primary: false
                }
            ]
        );
        
        document.body.appendChild(notification);
    }

    createNotification(title, message, actions = []) {
        const notification = document.createElement('div');
        notification.className = 'pwa-notification';
        
        notification.innerHTML = `
            <div class="pwa-notification-content">
                <div class="pwa-notification-header">
                    <h4>${title}</h4>
                    <button class="pwa-notification-close">&times;</button>
                </div>
                <p>${message}</p>
                <div class="pwa-notification-actions">
                    ${actions.map(action => `
                        <button class="pwa-notification-button ${action.primary ? 'primary' : 'secondary'}" 
                                data-action="${action.text}">
                            ${action.text}
                        </button>
                    `).join('')}
                </div>
            </div>
        `;
        
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(26, 26, 46, 0.95);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 212, 170, 0.3);
            border-radius: 12px;
            padding: 0;
            color: #ffffff;
            z-index: 1001;
            min-width: 300px;
            max-width: 400px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
            animation: slideIn 0.3s ease-out;
        `;
        
        // Add styles for internal elements
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            
            .pwa-notification-content {
                padding: 1rem;
            }
            
            .pwa-notification-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 0.5rem;
            }
            
            .pwa-notification-header h4 {
                margin: 0;
                color: #00d4aa;
                font-size: 1rem;
            }
            
            .pwa-notification-close {
                background: none;
                border: none;
                color: #ffffff;
                font-size: 1.5rem;
                cursor: pointer;
                padding: 0;
                width: 24px;
                height: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .pwa-notification-close:hover {
                color: #00d4aa;
            }
            
            .pwa-notification p {
                margin: 0 0 1rem 0;
                opacity: 0.9;
                line-height: 1.4;
            }
            
            .pwa-notification-actions {
                display: flex;
                gap: 0.5rem;
                justify-content: flex-end;
            }
            
            .pwa-notification-button {
                padding: 0.5rem 1rem;
                border: none;
                border-radius: 6px;
                font-size: 0.9rem;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .pwa-notification-button.primary {
                background: #00d4aa;
                color: #1a1a2e;
                font-weight: 600;
            }
            
            .pwa-notification-button.primary:hover {
                background: #00e5bb;
            }
            
            .pwa-notification-button.secondary {
                background: rgba(255, 255, 255, 0.1);
                color: #ffffff;
            }
            
            .pwa-notification-button.secondary:hover {
                background: rgba(255, 255, 255, 0.2);
            }
        `;
        
        document.head.appendChild(style);
        
        // Add event listeners
        notification.querySelector('.pwa-notification-close').addEventListener('click', () => {
            this.dismissNotification(notification);
        });
        
        actions.forEach(action => {
            const button = notification.querySelector(`[data-action="${action.text}"]`);
            button?.addEventListener('click', () => {
                action.action();
                this.dismissNotification(notification);
            });
        });
        
        // Auto-dismiss after 10 seconds
        setTimeout(() => {
            this.dismissNotification(notification);
        }, 10000);
        
        return notification;
    }

    dismissNotification(notification = null) {
        if (!notification) {
            notification = document.querySelector('.pwa-notification');
        }
        
        if (notification) {
            notification.style.animation = 'slideOut 0.3s ease-in forwards';
            setTimeout(() => {
                notification.remove();
            }, 300);
        }
    }

    async applyUpdate() {
        if (!this.registration || !this.registration.waiting) {
            console.warn('[PWA] No update available');
            return;
        }

        // Tell the waiting service worker to take control
        this.registration.waiting.postMessage({ type: 'SKIP_WAITING' });
        
        // Reload the page to apply the update
        window.location.reload();
    }

    showInstallSuccessMessage() {
        const notification = this.createNotification(
            'App Installed!',
            'LeanVibe Agent Hive has been installed successfully. You can now access it from your home screen.',
            [
                {
                    text: 'Great!',
                    action: () => {},
                    primary: true
                }
            ]
        );
        
        document.body.appendChild(notification);
    }

    showUpdateSuccessMessage() {
        const notification = this.createNotification(
            'Update Complete',
            'The app has been updated to the latest version.',
            [
                {
                    text: 'OK',
                    action: () => {},
                    primary: true
                }
            ]
        );
        
        document.body.appendChild(notification);
    }

    showOfflineReadyMessage() {
        const notification = this.createNotification(
            'Offline Ready',
            'The app is now available offline. You can continue using it even without an internet connection.',
            [
                {
                    text: 'Got it',
                    action: () => {},
                    primary: true
                }
            ]
        );
        
        document.body.appendChild(notification);
    }

    showInstallDeclinedMessage() {
        const notification = this.createNotification(
            'Install Later',
            'You can install the app anytime from your browser menu or by clicking the install button.',
            [
                {
                    text: 'OK',
                    action: () => {},
                    primary: true
                }
            ]
        );
        
        document.body.appendChild(notification);
    }

    showManualInstallInstructions() {
        const userAgent = navigator.userAgent.toLowerCase();
        let instructions = '';
        
        if (userAgent.includes('chrome')) {
            instructions = 'Click the install icon in the address bar or go to Chrome menu > Install app.';
        } else if (userAgent.includes('firefox')) {
            instructions = 'This app can be added to your home screen through your browser settings.';
        } else if (userAgent.includes('safari')) {
            instructions = 'Tap the share button and select "Add to Home Screen".';
        } else {
            instructions = 'Check your browser menu for "Install app" or "Add to home screen" option.';
        }
        
        const notification = this.createNotification(
            'Manual Installation',
            instructions,
            [
                {
                    text: 'Got it',
                    action: () => {},
                    primary: true
                }
            ]
        );
        
        document.body.appendChild(notification);
    }

    // Public API
    isAppInstalled() {
        return this.isInstalled;
    }

    isUpdateAvailable() {
        return this.updateAvailable;
    }

    async checkForUpdates() {
        if (this.registration) {
            await this.registration.update();
        }
    }

    async requestNotificationPermission() {
        if ('Notification' in window) {
            const permission = await Notification.requestPermission();
            console.log('[PWA] Notification permission:', permission);
            return permission === 'granted';
        }
        return false;
    }

    async enablePushNotifications() {
        if (!('serviceWorker' in navigator) || !('PushManager' in window)) {
            console.warn('[PWA] Push notifications not supported');
            return false;
        }

        try {
            const subscription = await this.registration.pushManager.subscribe({
                userVisibleOnly: true,
                applicationServerKey: this.urlBase64ToUint8Array('YOUR_VAPID_PUBLIC_KEY')
            });
            
            console.log('[PWA] Push subscription:', subscription);
            return subscription;
        } catch (error) {
            console.error('[PWA] Push subscription failed:', error);
            return false;
        }
    }

    urlBase64ToUint8Array(base64String) {
        const padding = '='.repeat((4 - base64String.length % 4) % 4);
        const base64 = (base64String + padding)
            .replace(/-/g, '+')
            .replace(/_/g, '/');

        const rawData = window.atob(base64);
        const outputArray = new Uint8Array(rawData.length);

        for (let i = 0; i < rawData.length; ++i) {
            outputArray[i] = rawData.charCodeAt(i);
        }
        return outputArray;
    }
}

// Initialize PWA Manager
const pwaManager = new PWAManager();

// Export for use in other modules
export { pwaManager };