/**
 * Agentic UX - Background Service Worker
 * Aggregates behavioral data from content scripts and communicates with Python backend agents
 * Implements message routing, coordination protocols, and intelligent batching
 */

class BehavioralDataAggregator {
  constructor() {
    this.config = {
      backendUrl: 'ws://localhost:8765',
      maxBufferSize: 10000,
      batchSize: 100,
      batchTimeoutMs: 5000,
      reconnectDelayMs: 3000,
      maxReconnectAttempts: 10,
      enableCompression: true,
      enableEncryption: false,
      dataRetentionMs: 3600000, // 1 hour
      privacyMode: false
    };

    this.state = {
      wsConnected: false,
      reconnectAttempts: 0,
      isProcessing: false,
      activeTabs: new Map(),
      sessionId: this.generateSessionId(),
      startTime: Date.now()
    };

    this.dataStore = {
      pendingBatch: [],
      processedCount: 0,
      errorCount: 0,
      lastProcessTime: Date.now(),
      tabSessions: new Map()
    };

    this.routingTable = new Map();
    this.messageQueue = [];
    this.batchTimer = null;

    this.setupStorageListener();
    this.setupTabListeners();
    this.setupMessageListeners();
    this.initializeWebSocket();
    this.startBatchProcessor();
  }

  /**
   * Setup Chrome storage listener for persistence
   */
  setupStorageListener() {
    chrome.storage.onChanged.addListener((changes, area) => {
      if (area === 'local') {
        // Handle storage changes
        if (changes.config) {
          Object.assign(this.config, changes.config.newValue);
          this.updateWebSocketConfig();
        }
      }
    });
  }

  /**
   * Setup tab lifecycle listeners
   */
  setupTabListeners() {
    chrome.tabs.onActivated.addListener((activeInfo) => {
      this.handleTabActivated(activeInfo.tabId);
    });

    chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
      if (changeInfo.status === 'complete') {
        this.handleTabLoaded(tabId, tab);
      }
    });

    chrome.tabs.onRemoved.addListener((tabId) => {
      this.handleTabClosed(tabId);
    });
  }

  /**
   * Setup message listener for content scripts and internal messages
   */
  setupMessageListeners() {
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      this.routeMessage(request, sender, sendResponse);
      return true; // Keep channel open for async response
    });

    chrome.runtime.onMessageExternal?.addListener?.((request, sender, sendResponse) => {
      // Handle messages from other extensions (if needed)
      this.handleExternalMessage(request, sender, sendResponse);
      return true;
    });
  }

  /**
   * Route incoming messages to appropriate handlers
   */
  routeMessage(request, sender, sendResponse) {
    try {
      const { type, eventType, data, pageUrl, timestamp } = request;

      if (type === 'BEHAVIORAL_DATA') {
        this.processBehavioralData({
          eventType,
          data,
          pageUrl,
          timestamp,
          tabId: sender.tab.id,
          frameId: sender.frameId
        }, sendResponse);
      } else if (type === 'AGENT_REQUEST') {
        this.handleAgentRequest(request, sendResponse);
      } else if (type === 'SYNC_REQUEST') {
        this.handleSyncRequest(sender.tab.id, sendResponse);
      } else if (type === 'HEALTH_CHECK') {
        sendResponse({ status: 'healthy', wsConnected: this.state.wsConnected });
      } else {
        console.warn('[Agentic UX] Unknown message type:', type);
        sendResponse({ error: 'Unknown message type' });
      }
    } catch (error) {
      console.error('[Agentic UX] Error routing message:', error);
      sendResponse({ error: error.message });
    }
  }

  /**
   * Process behavioral data from content scripts
   */
  processBehavioralData(behavioralEvent, sendResponse) {
    try {
      const { eventType, data, pageUrl, timestamp, tabId } = behavioralEvent;

      // Track tab sessions
      if (!this.dataStore.tabSessions.has(tabId)) {
        this.dataStore.tabSessions.set(tabId, {
          startTime: Date.now(),
          events: [],
          pageUrl: pageUrl
        });
      }

      const tabSession = this.dataStore.tabSessions.get(tabId);
      tabSession.events.push({
        eventType,
        data,
        timestamp,
        tabId,
        pageUrl
      });

      // Add to pending batch
      if (this.dataStore.pendingBatch.length < this.config.maxBufferSize) {
        this.dataStore.pendingBatch.push({
          type: 'BEHAVIORAL_EVENT',
          eventType,
          payload: {
            data,
            pageUrl,
            timestamp,
            tabId,
            sessionId: this.state.sessionId
          },
          sequenceNumber: this.dataStore.pendingBatch.length
        });

        // Trigger batch if threshold reached
        if (this.dataStore.pendingBatch.length >= this.config.batchSize) {
          this.processPendingBatch();
        }
      } else {
        console.warn('[Agentic UX] Pending batch size limit reached');
        this.dataStore.errorCount++;
      }

      sendResponse({ success: true, buffered: this.dataStore.pendingBatch.length });
    } catch (error) {
      console.error('[Agentic UX] Error processing behavioral data:', error);
      this.dataStore.errorCount++;
      sendResponse({ success: false, error: error.message });
    }
  }

  /**
   * Process pending batch of behavioral data
   */
  processPendingBatch() {
    if (this.dataStore.isProcessing || this.dataStore.pendingBatch.length === 0) {
      return;
    }

    this.dataStore.isProcessing = true;

    try {
      const batch = this.dataStore.pendingBatch.splice(0, this.config.batchSize);

      // Optionally compress batch
      let payload = batch;
      if (this.config.enableCompression) {
        payload = this.compressBatch(batch);
      }

      // Create protocol message
      const message = {
        type: 'BEHAVIORAL_BATCH',
        sessionId: this.state.sessionId,
        timestamp: Date.now(),
        batchSize: batch.length,
        payload: payload,
        metadata: {
          extensionVersion: chrome.runtime.getManifest().version,
          tabCount: this.state.activeTabs.size,
          uptime: Date.now() - this.state.startTime
        }
      };

      // Send to WebSocket or queue
      if (this.state.wsConnected) {
        this.sendToBackend(message);
      } else {
        this.messageQueue.push(message);
        if (this.messageQueue.length > 100) {
          this.messageQueue.shift(); // Discard oldest if queue too large
        }
      }

      this.dataStore.processedCount += batch.length;
      this.dataStore.lastProcessTime = Date.now();
    } catch (error) {
      console.error('[Agentic UX] Error processing batch:', error);
      this.dataStore.errorCount++;
    } finally {
      this.dataStore.isProcessing = false;
    }
  }

  /**
   * Start batch processor timer
   */
  startBatchProcessor() {
    if (this.batchTimer) clearInterval(this.batchTimer);

    this.batchTimer = setInterval(() => {
      if (this.dataStore.pendingBatch.length > 0) {
        this.processPendingBatch();
      }

      // Clean up old tab sessions
      const now = Date.now();
      for (const [tabId, session] of this.dataStore.tabSessions) {
        if (now - session.startTime > this.config.dataRetentionMs) {
          this.dataStore.tabSessions.delete(tabId);
        }
      }
    }, this.config.batchTimeoutMs);
  }

  /**
   * Compress batch data (simple delta encoding)
   */
  compressBatch(batch) {
    try {
      // Create a simplified representation to reduce data size
      const compressed = {
        events: batch.length,
        types: [...new Set(batch.map(e => e.eventType))],
        timeRange: {
          start: batch[0]?.payload?.timestamp,
          end: batch[batch.length - 1]?.payload?.timestamp
        },
        data: batch
      };
      return compressed;
    } catch (error) {
      console.error('[Agentic UX] Error compressing batch:', error);
      return batch;
    }
  }

  /**
   * Initialize WebSocket connection to Python backend
   */
  initializeWebSocket() {
    try {
      // Check if WebSocket is available
      if (typeof WebSocket === 'undefined') {
        console.warn('[Agentic UX] WebSocket not available in background worker');
        return;
      }

      this.ws = new WebSocket(this.config.backendUrl);

      this.ws.onopen = () => this.handleWebSocketOpen();
      this.ws.onmessage = (event) => this.handleWebSocketMessage(event);
      this.ws.onerror = (error) => this.handleWebSocketError(error);
      this.ws.onclose = () => this.handleWebSocketClose();
    } catch (error) {
      console.error('[Agentic UX] Error initializing WebSocket:', error);
      this.scheduleReconnect();
    }
  }

  /**
   * Handle WebSocket connection opened
   */
  handleWebSocketOpen() {
    console.log('[Agentic UX] WebSocket connected');
    this.state.wsConnected = true;
    this.state.reconnectAttempts = 0;

    // Send handshake message
    const handshake = {
      type: 'HANDSHAKE',
      sessionId: this.state.sessionId,
      version: chrome.runtime.getManifest().version,
      platform: 'chrome-extension',
      timestamp: Date.now()
    };

    this.sendToBackend(handshake);

    // Flush message queue
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      this.sendToBackend(message);
    }
  }

  /**
   * Handle incoming WebSocket messages
   */
  handleWebSocketMessage(event) {
    try {
      const message = JSON.parse(event.data);

      if (message.type === 'AGENT_RESPONSE') {
        this.handleAgentResponse(message);
      } else if (message.type === 'COORDINATION') {
        this.handleCoordinationMessage(message);
      } else if (message.type === 'CONFIG_UPDATE') {
        this.handleConfigUpdate(message);
      } else if (message.type === 'HEARTBEAT') {
        // Respond to heartbeat
        this.sendToBackend({
          type: 'HEARTBEAT_ACK',
          sessionId: this.state.sessionId,
          timestamp: Date.now()
        });
      }
    } catch (error) {
      console.error('[Agentic UX] Error handling WebSocket message:', error);
    }
  }

  /**
   * Handle WebSocket errors
   */
  handleWebSocketError(error) {
    console.error('[Agentic UX] WebSocket error:', error);
    this.dataStore.errorCount++;
    this.scheduleReconnect();
  }

  /**
   * Handle WebSocket closure
   */
  handleWebSocketClose() {
    console.warn('[Agentic UX] WebSocket disconnected');
    this.state.wsConnected = false;
    this.scheduleReconnect();
  }

  /**
   * Schedule WebSocket reconnection with exponential backoff
   */
  scheduleReconnect() {
    if (this.state.reconnectAttempts >= this.config.maxReconnectAttempts) {
      console.error('[Agentic UX] Max reconnect attempts reached');
      return;
    }

    this.state.reconnectAttempts++;
    const delay = Math.min(
      this.config.reconnectDelayMs * Math.pow(2, this.state.reconnectAttempts - 1),
      30000 // Max 30 seconds
    );

    console.log(`[Agentic UX] Scheduling reconnect in ${delay}ms (attempt ${this.state.reconnectAttempts})`);

    setTimeout(() => {
      this.initializeWebSocket();
    }, delay);
  }

  /**
   * Send message to backend via WebSocket
   */
  sendToBackend(message) {
    try {
      if (this.state.wsConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify(message));
      } else {
        this.messageQueue.push(message);
      }
    } catch (error) {
      console.error('[Agentic UX] Error sending to backend:', error);
      this.messageQueue.push(message);
    }
  }

  /**
   * Handle agent response from backend
   */
  handleAgentResponse(message) {
    const { payload, targetTabId, messageId } = message;

    if (targetTabId) {
      // Route response to specific tab
      chrome.tabs.sendMessage(targetTabId, {
        type: 'AGENT_RESPONSE',
        payload,
        messageId
      }).catch(error => {
        console.warn('[Agentic UX] Error sending response to tab:', error);
      });
    }

    // Store response in local storage for persistence
    chrome.storage.local.set({
      [`agentResponse_${messageId}`]: {
        payload,
        timestamp: Date.now()
      }
    }).catch(error => {
      console.error('[Agentic UX] Error storing response:', error);
    });
  }

  /**
   * Handle coordination messages from backend
   */
  handleCoordinationMessage(message) {
    const { action, data } = message;

    switch (action) {
      case 'BROADCAST_TO_TABS':
        this.broadcastToAllTabs(data);
        break;
      case 'UPDATE_TAB_CONFIG':
        this.updateTabConfig(data.tabId, data.config);
        break;
      case 'PAUSE_COLLECTION':
        this.pauseDataCollection();
        break;
      case 'RESUME_COLLECTION':
        this.resumeDataCollection();
        break;
      default:
        console.warn('[Agentic UX] Unknown coordination action:', action);
    }
  }

  /**
   * Handle config updates from backend
   */
  handleConfigUpdate(message) {
    try {
      Object.assign(this.config, message.config);
      chrome.storage.local.set({ config: this.config });
      console.log('[Agentic UX] Configuration updated:', this.config);
    } catch (error) {
      console.error('[Agentic UX] Error updating config:', error);
    }
  }

  /**
   * Broadcast message to all active tabs
   */
  broadcastToAllTabs(message) {
    chrome.tabs.query({}, (tabs) => {
      tabs.forEach(tab => {
        chrome.tabs.sendMessage(tab.id, message).catch(error => {
          // Ignore errors for inactive tabs
        });
      });
    });
  }

  /**
   * Update configuration for specific tab
   */
  updateTabConfig(tabId, config) {
    chrome.tabs.sendMessage(tabId, {
      type: 'UPDATE_CONFIG',
      config
    }).catch(error => {
      console.warn('[Agentic UX] Error updating tab config:', error);
    });
  }

  /**
   * Pause data collection across all tabs
   */
  pauseDataCollection() {
    this.broadcastToAllTabs({
      type: 'SET_ACTIVE',
      active: false
    });
    console.log('[Agentic UX] Data collection paused');
  }

  /**
   * Resume data collection across all tabs
   */
  resumeDataCollection() {
    this.broadcastToAllTabs({
      type: 'SET_ACTIVE',
      active: true
    });
    console.log('[Agentic UX] Data collection resumed');
  }

  /**
   * Handle agent request from content script
   */
  handleAgentRequest(request, sendResponse) {
    const { action, payload, tabId } = request;

    const agentMessage = {
      type: 'AGENT_REQUEST',
      action,
      payload,
      sourceTabId: tabId,
      sessionId: this.state.sessionId,
      timestamp: Date.now(),
      messageId: this.generateMessageId()
    };

    this.sendToBackend(agentMessage);

    // Send response immediately, actual response comes via AGENT_RESPONSE
    sendResponse({
      messageId: agentMessage.messageId,
      queued: true
    });
  }

  /**
   * Handle sync request from content script
   */
  handleSyncRequest(tabId, sendResponse) {
    const tabSession = this.dataStore.tabSessions.get(tabId);

    sendResponse({
      success: true,
      session: tabSession || null,
      stats: {
        totalProcessed: this.dataStore.processedCount,
        pendingBatch: this.dataStore.pendingBatch.length,
        errors: this.dataStore.errorCount,
        wsConnected: this.state.wsConnected
      }
    });
  }

  /**
   * Handle tab activation
   */
  handleTabActivated(tabId) {
    this.state.activeTabs.set(tabId, Date.now());

    chrome.tabs.get(tabId, (tab) => {
      if (!chrome.runtime.lastError) {
        // Update tab session
        if (this.dataStore.tabSessions.has(tabId)) {
          const session = this.dataStore.tabSessions.get(tabId);
          session.lastActive = Date.now();
        }
      }
    });
  }

  /**
   * Handle tab loaded
   */
  handleTabLoaded(tabId, tab) {
    if (this.dataStore.tabSessions.has(tabId)) {
      const session = this.dataStore.tabSessions.get(tabId);
      session.pageUrl = tab.url;
      session.reloaded = Date.now();
    }
  }

  /**
   * Handle tab closed
   */
  handleTabClosed(tabId) {
    const session = this.dataStore.tabSessions.get(tabId);
    if (session) {
      // Send final session data before removing
      this.sendToBackend({
        type: 'SESSION_CLOSED',
        sessionId: this.state.sessionId,
        tabId,
        sessionData: session,
        timestamp: Date.now()
      });

      this.dataStore.tabSessions.delete(tabId);
    }

    this.state.activeTabs.delete(tabId);
  }

  /**
   * Update WebSocket configuration
   */
  updateWebSocketConfig() {
    if (this.ws) {
      this.ws.close();
      this.state.wsConnected = false;
      setTimeout(() => this.initializeWebSocket(), 1000);
    }
  }

  /**
   * Generate unique session ID
   */
  generateSessionId() {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Generate unique message ID
   */
  generateMessageId() {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Handle external messages (from other extensions)
   */
  handleExternalMessage(request, sender, sendResponse) {
    if (request.type === 'GET_SESSION_INFO') {
      sendResponse({
        sessionId: this.state.sessionId,
        wsConnected: this.state.wsConnected,
        activeTabs: this.state.activeTabs.size
      });
    } else {
      sendResponse({ error: 'Unknown external message' });
    }
  }

  /**
   * Get extension status for debugging
   */
  getStatus() {
    return {
      sessionId: this.state.sessionId,
      wsConnected: this.state.wsConnected,
      reconnectAttempts: this.state.reconnectAttempts,
      activeTabs: this.state.activeTabs.size,
      pendingBatch: this.dataStore.pendingBatch.length,
      processedCount: this.dataStore.processedCount,
      errorCount: this.dataStore.errorCount,
      uptime: Date.now() - this.state.startTime,
      lastProcessTime: this.dataStore.lastProcessTime
    };
  }
}

// Initialize the aggregator
const aggregator = new BehavioralDataAggregator();

// Command handler for extension actions
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    console.log('[Agentic UX] Extension installed');
    // Open onboarding page if needed
  } else if (details.reason === 'update') {
    console.log('[Agentic UX] Extension updated');
  }
});

// Expose status via chrome.runtime API for debugging
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === 'GET_STATUS') {
    sendResponse(aggregator.getStatus());
    return true;
  }
});

console.log('[Agentic UX] Background service worker initialized');
