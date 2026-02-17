/**
 * Agentic UX - Content Script
 * Captures user behavioral patterns and sends them to the background service worker
 * Optimized for performance with minimal overhead on page rendering
 */

class CircularBuffer {
  constructor(maxSize = 1000) {
    this.maxSize = maxSize;
    this.buffer = [];
    this.index = 0;
  }

  add(item) {
    if (this.buffer.length < this.maxSize) {
      this.buffer.push(item);
    } else {
      this.buffer[this.index] = item;
    }
    this.index = (this.index + 1) % this.maxSize;
  }

  get() {
    return this.buffer.slice(0, this.buffer.length);
  }

  clear() {
    this.buffer = [];
    this.index = 0;
  }

  size() {
    return this.buffer.length;
  }
}

class BehavioralDataCollector {
  constructor() {
    this.config = {
      mouseMovementSampleRate: 100,
      scrollEventDebounce: 150,
      keyEventDebounce: 50,
      focusTrackingEnabled: true,
      navigationTrackingEnabled: true,
      performanceMetricsEnabled: true,
      privacyMode: false,
      dataMinimization: true
    };

    this.buffers = {
      mouseMovements: new CircularBuffer(500),
      clicks: new CircularBuffer(200),
      scrollEvents: new CircularBuffer(200),
      keyEvents: new CircularBuffer(100),
      focusEvents: new CircularBuffer(100),
      navigationEvents: new CircularBuffer(50),
      performanceMetrics: new CircularBuffer(100)
    };

    this.state = {
      isActive: true,
      lastMouseMove: 0,
      lastScroll: 0,
      lastKeyEvent: 0,
      sessionStartTime: Date.now(),
      pageStartTime: Date.now(),
      totalMouseMovements: 0,
      totalClicks: 0,
      totalScrolls: 0,
      totalKeyEvents: 0,
      activeElementTracker: new Map(),
      pageVisibility: document.visibilityState
    };

    this.elementMetadata = new WeakMap();
    this.setupEventListeners();
    this.startPeriodicSync();
  }

  /**
   * Setup all event listeners with performance optimizations
   */
  setupEventListeners() {
    try {
      // Mouse movement tracking with throttling
      document.addEventListener('mousemove', this.throttle(
        (e) => this.handleMouseMove(e),
        this.config.mouseMovementSampleRate
      ), { passive: true, capture: false });

      // Click tracking
      document.addEventListener('click', (e) => this.handleClick(e), {
        passive: true,
        capture: true
      });

      // Scroll tracking with debouncing
      window.addEventListener('scroll', this.debounce(
        () => this.handleScroll(),
        this.config.scrollEventDebounce
      ), { passive: true, capture: false });

      // Keyboard event tracking with debouncing
      document.addEventListener('keydown', this.debounce(
        (e) => this.handleKeyEvent(e),
        this.config.keyEventDebounce
      ), { passive: true, capture: true });

      // Focus change tracking
      if (this.config.focusTrackingEnabled) {
        document.addEventListener('focusin', (e) => this.handleFocusChange(e, 'focus'),
          { passive: true, capture: true });
        document.addEventListener('focusout', (e) => this.handleFocusChange(e, 'blur'),
          { passive: true, capture: true });
      }

      // Navigation tracking
      if (this.config.navigationTrackingEnabled) {
        window.addEventListener('beforeunload', () => this.handleNavigation('unload'));
        window.addEventListener('pagehide', () => this.handleNavigation('pagehide'));
        window.addEventListener('pageshow', () => this.handleNavigation('pageshow'));
      }

      // Page visibility tracking
      document.addEventListener('visibilitychange', () => {
        this.state.pageVisibility = document.visibilityState;
        this.sendBehavioralEvent('visibility_change', {
          visibility: document.visibilityState,
          timestamp: Date.now()
        });
      }, { passive: true });

      // Performance observer
      if (this.config.performanceMetricsEnabled && 'PerformanceObserver' in window) {
        this.setupPerformanceObserver();
      }

    } catch (error) {
      console.error('[Agentic UX] Error setting up event listeners:', error);
    }
  }

  /**
   * Setup performance observer for page metrics
   */
  setupPerformanceObserver() {
    try {
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (entry.entryType === 'largest-contentful-paint' ||
              entry.entryType === 'first-input' ||
              entry.entryType === 'layout-shift') {
            this.buffers.performanceMetrics.add({
              type: entry.entryType,
              value: entry.value || entry.duration,
              timestamp: Date.now()
            });
          }
        }
      });

      observer.observe({
        entryTypes: ['largest-contentful-paint', 'first-input', 'layout-shift']
      });
    } catch (error) {
      console.debug('[Agentic UX] PerformanceObserver not available:', error.message);
    }
  }

  /**
   * Handle mouse movement events
   */
  handleMouseMove(event) {
    if (!this.state.isActive) return;

    const now = Date.now();
    if (now - this.state.lastMouseMove < this.config.mouseMovementSampleRate) {
      return;
    }

    this.state.lastMouseMove = now;
    this.state.totalMouseMovements++;

    // Privacy-preserving: don't capture absolute coordinates, use viewport-relative
    const viewportX = event.clientX;
    const viewportY = event.clientY;
    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;

    // Normalize to 0-1 range for privacy
    const normalizedX = viewportX / viewportWidth;
    const normalizedY = viewportY / viewportHeight;

    // Determine if movement is near interactive elements
    const targetElement = document.elementFromPoint(viewportX, viewportY);
    const isNearInteractive = this.isInteractiveElement(targetElement);

    this.buffers.mouseMovements.add({
      type: 'mouse_move',
      normalizedX: Math.round(normalizedX * 1000) / 1000,
      normalizedY: Math.round(normalizedY * 1000) / 1000,
      nearInteractive: isNearInteractive,
      timestamp: now,
      velocity: this.calculateMouseVelocity()
    });
  }

  /**
   * Handle click events
   */
  handleClick(event) {
    if (!this.state.isActive) return;

    this.state.totalClicks++;
    const target = event.target;

    // Privacy: Don't capture sensitive input values
    const elementType = target.tagName.toLowerCase();
    let elementInfo = {
      tag: elementType,
      hasClass: target.className ? target.className.split(' ').length : 0,
      isButton: elementType === 'button',
      isLink: elementType === 'a',
      isInput: elementType === 'input'
    };

    // For input fields, only track type, not value
    if (target.type) {
      elementInfo.inputType = target.type;
    }

    const clickData = {
      type: 'click',
      element: elementInfo,
      position: {
        normalizedX: Math.round((event.clientX / window.innerWidth) * 1000) / 1000,
        normalizedY: Math.round((event.clientY / window.innerHeight) * 1000) / 1000
      },
      button: event.button,
      timestamp: Date.now(),
      ctrlKey: event.ctrlKey,
      shiftKey: event.shiftKey
    };

    this.buffers.clicks.add(clickData);
  }

  /**
   * Handle scroll events
   */
  handleScroll() {
    if (!this.state.isActive) return;

    const now = Date.now();
    if (now - this.state.lastScroll < this.config.scrollEventDebounce) {
      return;
    }

    this.state.lastScroll = now;
    this.state.totalScrolls++;

    const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
    const scrollTop = window.scrollY;
    const scrollPercentage = scrollHeight > 0 ? (scrollTop / scrollHeight) : 0;

    this.buffers.scrollEvents.add({
      type: 'scroll',
      scrollPercentage: Math.round(scrollPercentage * 1000) / 1000,
      scrollDirection: this.calculateScrollDirection(),
      timestamp: now,
      depth: Math.round(scrollTop / window.innerHeight)
    });
  }

  /**
   * Handle keyboard events
   */
  handleKeyEvent(event) {
    if (!this.state.isActive) return;

    const now = Date.now();
    if (now - this.state.lastKeyEvent < this.config.keyEventDebounce) {
      return;
    }

    this.state.lastKeyEvent = now;
    this.state.totalKeyEvents++;

    // Privacy: Don't capture actual key values for sensitive inputs
    const target = event.target;
    const isSensitiveInput = this.isSensitiveInput(target);

    let keyData = {
      type: 'key_event',
      eventType: event.type,
      timestamp: now,
      target: {
        tag: target.tagName.toLowerCase(),
        isSensitive: isSensitiveInput
      }
    };

    // Only track non-sensitive keys for non-sensitive inputs
    if (!isSensitiveInput && !this.isPrivateKey(event.key)) {
      keyData.key = event.key.length === 1 ? 'character' : event.key;
      keyData.ctrlKey = event.ctrlKey;
      keyData.shiftKey = event.shiftKey;
    }

    this.buffers.keyEvents.add(keyData);
  }

  /**
   * Handle focus change events
   */
  handleFocusChange(event, type) {
    if (!this.state.isActive) return;

    const target = event.target;
    const elementType = target.tagName.toLowerCase();

    const focusData = {
      type: 'focus_change',
      eventType: type,
      element: {
        tag: elementType,
        role: target.getAttribute('role'),
        placeholder: target.placeholder ? 'present' : 'absent'
      },
      timestamp: Date.now()
    };

    // Track focus history for interaction patterns
    if (type === 'focus') {
      const elementKey = `${elementType}_${target.id || 'unnamed'}`;
      this.state.activeElementTracker.set(elementKey, Date.now());
    }

    this.buffers.focusEvents.add(focusData);
  }

  /**
   * Handle navigation events
   */
  handleNavigation(type) {
    if (!this.config.navigationTrackingEnabled) return;

    const navigationData = {
      type: 'navigation',
      eventType: type,
      url: window.location.href,
      timestamp: Date.now(),
      sessionDuration: Date.now() - this.state.pageStartTime,
      interactionCount: this.state.totalClicks + this.state.totalMouseMovements
    };

    this.buffers.navigationEvents.add(navigationData);

    // Send final batch of data before navigation
    this.sendBehavioralEvent('navigation', navigationData);
  }

  /**
   * Utility: Throttle function for high-frequency events
   */
  throttle(func, limit) {
    let inThrottle;
    return function(...args) {
      if (!inThrottle) {
        func.apply(this, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  }

  /**
   * Utility: Debounce function for events
   */
  debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  /**
   * Determine if element is interactive
   */
  isInteractiveElement(element) {
    if (!element) return false;
    const interactiveTags = ['button', 'a', 'input', 'select', 'textarea', 'label'];
    const roles = ['button', 'link', 'menuitem', 'tab', 'checkbox', 'radio'];

    return interactiveTags.includes(element.tagName.toLowerCase()) ||
           roles.includes(element.getAttribute('role'));
  }

  /**
   * Check if input is sensitive (password, credit card, etc.)
   */
  isSensitiveInput(target) {
    if (target.tagName.toLowerCase() !== 'input') return false;
    const sensitiveTypes = ['password', 'email', 'tel', 'number', 'credit-card'];
    return sensitiveTypes.includes(target.type) || target.name?.includes('password');
  }

  /**
   * Check if key press should not be recorded (passwords, etc.)
   */
  isPrivateKey(key) {
    return ['Enter', 'Tab', 'Escape'].includes(key);
  }

  /**
   * Calculate mouse velocity for gesture recognition
   */
  calculateMouseVelocity() {
    // Simplified velocity calculation (would be enhanced in production)
    return 'normal';
  }

  /**
   * Calculate scroll direction
   */
  calculateScrollDirection() {
    // Enhanced with actual scroll tracking in production
    return window.scrollY > 0 ? 'down' : 'up';
  }

  /**
   * Send behavioral event to background worker
   */
  sendBehavioralEvent(eventType, data) {
    try {
      chrome.runtime.sendMessage({
        type: 'BEHAVIORAL_DATA',
        eventType: eventType,
        data: data,
        pageUrl: window.location.href,
        timestamp: Date.now()
      }, (response) => {
        if (chrome.runtime.lastError) {
          console.debug('[Agentic UX] Background worker not available:', chrome.runtime.lastError.message);
        }
      });
    } catch (error) {
      console.error('[Agentic UX] Error sending behavioral event:', error);
    }
  }

  /**
   * Periodic sync to send accumulated behavioral data
   */
  startPeriodicSync() {
    setInterval(() => {
      if (!this.state.isActive) return;

      const behavioralSnapshot = {
        mouseMovements: this.buffers.mouseMovements.size(),
        clicks: this.buffers.clicks.size(),
        scrollEvents: this.buffers.scrollEvents.size(),
        keyEvents: this.buffers.keyEvents.size(),
        focusEvents: this.buffers.focusEvents.size(),
        performanceMetrics: this.buffers.performanceMetrics.size(),
        sessionMetrics: {
          totalMouseMovements: this.state.totalMouseMovements,
          totalClicks: this.state.totalClicks,
          totalScrolls: this.state.totalScrolls,
          totalKeyEvents: this.state.totalKeyEvents,
          sessionDuration: Date.now() - this.state.sessionStartTime,
          pageDuration: Date.now() - this.state.pageStartTime
        }
      };

      this.sendBehavioralEvent('periodic_sync', behavioralSnapshot);
    }, 5000); // Sync every 5 seconds
  }

  /**
   * Get all buffered data (for debugging)
   */
  getAllData() {
    return {
      mouseMovements: this.buffers.mouseMovements.get(),
      clicks: this.buffers.clicks.get(),
      scrollEvents: this.buffers.scrollEvents.get(),
      keyEvents: this.buffers.keyEvents.get(),
      focusEvents: this.buffers.focusEvents.get(),
      navigationEvents: this.buffers.navigationEvents.get(),
      performanceMetrics: this.buffers.performanceMetrics.get(),
      state: this.state
    };
  }

  /**
   * Control extension activation/deactivation
   */
  setActive(active) {
    this.state.isActive = active;
    this.sendBehavioralEvent('activation_change', { active });
  }

  /**
   * Clear all buffers (for memory management)
   */
  clearBuffers() {
    Object.values(this.buffers).forEach(buffer => buffer.clear());
  }
}

// Initialize the behavioral data collector
const collector = new BehavioralDataCollector();

// Listen for messages from background worker
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  try {
    if (request.type === 'GET_BEHAVIORAL_DATA') {
      sendResponse({
        success: true,
        data: collector.getAllData()
      });
    } else if (request.type === 'SET_ACTIVE') {
      collector.setActive(request.active);
      sendResponse({ success: true });
    } else if (request.type === 'CLEAR_BUFFERS') {
      collector.clearBuffers();
      sendResponse({ success: true });
    } else if (request.type === 'GET_CONFIG') {
      sendResponse({
        success: true,
        config: collector.config
      });
    } else if (request.type === 'UPDATE_CONFIG') {
      Object.assign(collector.config, request.config);
      sendResponse({ success: true });
    }
  } catch (error) {
    console.error('[Agentic UX] Error handling message:', error);
    sendResponse({ success: false, error: error.message });
  }
});

console.log('[Agentic UX] Content script loaded and behavioral tracking initialized');
