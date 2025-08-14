/* Shared Sidebar Component JavaScript - RDL Migration Tool */

// Restore state immediately before DOM content loads to prevent flashing
(function() {
    const isCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
    if (isCollapsed) {
        document.documentElement.style.setProperty('--sidebar-initial-width', '70px');
        document.documentElement.style.setProperty('--main-initial-margin', '70px');
    } else {
        document.documentElement.style.setProperty('--sidebar-initial-width', '280px');
        document.documentElement.style.setProperty('--main-initial-margin', '280px');
    }
})();

// Sidebar management class
class SidebarManager {
    constructor() {
        this.sidebar = document.getElementById('sidebar');
        this.mainContent = document.getElementById('mainContent');
        this.sidebarToggle = document.getElementById('sidebarToggle');
        this.init();
    }
    
    init() {
        this.restoreSidebarState();
        this.bindEvents();
    }
    
    // restore sidebar state from localStorage
    restoreSidebarState() {
        const isCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
        if (isCollapsed) {
            this.sidebar?.classList.add('collapsed');
            this.mainContent?.classList.add('expanded');
        }
    }
    
    // save sidebar state to localStorage
    saveSidebarState() {
        const isCollapsed = this.sidebar?.classList.contains('collapsed');
        localStorage.setItem('sidebarCollapsed', isCollapsed);
    }
    
    // toggle functionality
    toggleSidebar() {
        if (window.innerWidth <= 768) {
            // mobile behavior - show/hide sidebar
            this.sidebar?.classList.toggle('show');
        } else {
            // desktop behavior - collapse/expand sidebar
            this.sidebar?.classList.toggle('collapsed');
            this.mainContent?.classList.toggle('expanded');
        }
        
        // save state
        this.saveSidebarState();
    }
    
    // bind event listeners
    bindEvents() {
        try {
            if (this.sidebarToggle) {
                this.sidebarToggle.addEventListener('click', () => {
                    this.toggleSidebar();
                });
            }
        } catch (error) {
            console.error('Sidebar event listener error:', error);
        }
    }
}

// Initialize sidebar when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('🎯 Initializing Shared Sidebar Manager');
    new SidebarManager();
});