package api

import (
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/labstack/echo/v4"
	"github.com/mumax/3/engine"
	"github.com/mumax/3/logUI"
	"github.com/vmihailenco/msgpack/v5"
)

// WebSocketManager holds state previously stored in global variables.
type WebSocketManager struct {
	upgrader       websocket.Upgrader
	connections    *connectionManager
	lastStep       int
	broadcastStop  chan struct{}
	broadcastStart sync.Once
	engineState    *EngineState
}

type connectionManager struct {
	conns map[*websocket.Conn]struct{}
	mu    sync.Mutex
}

func newWebSocketManager() *WebSocketManager {
	return &WebSocketManager{
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				return true
			},
		},
		connections: &connectionManager{
			conns: make(map[*websocket.Conn]struct{}),
			mu:    sync.Mutex{},
		},
		broadcastStop: make(chan struct{}),
	}
}

func (cm *connectionManager) add(ws *websocket.Conn) {
	logUI.Log.Debug("Websocket connection added")
	cm.mu.Lock()
	defer cm.mu.Unlock()
	cm.conns[ws] = struct{}{}
}

func (cm *connectionManager) remove(ws *websocket.Conn) {
	logUI.Log.Debug("Websocket connection removed")
	cm.mu.Lock()
	defer cm.mu.Unlock()
	delete(cm.conns, ws)
}

func (cm *connectionManager) broadcast(msg []byte) {
	cm.mu.Lock()
	defer cm.mu.Unlock()
	for ws := range cm.conns {
		err := ws.WriteMessage(websocket.BinaryMessage, msg)
		if err != nil {
			logUI.Log.Err("Error sending message via WebSocket: %v", err)
			ws.Close()
			delete(cm.conns, ws)
		}
	}
}

func (wsManager *WebSocketManager) websocketEntrypoint(c echo.Context) error {
	logUI.Log.Debug("New WebSocket connection, upgrading...")
	ws, err := wsManager.upgrader.Upgrade(c.Response(), c.Request(), nil)
	logUI.Log.Debug("New WebSocket connection upgraded")
	if err != nil {
		logUI.Log.Err("Error upgrading connection to WebSocket: %v", err)
		return err
	}
	defer ws.Close()

	wsManager.connections.add(ws)
	defer wsManager.connections.remove(ws)
	wsManager.engineState.Preview.Refresh = true
	wsManager.broadcastEngineState()

	// Channel to signal when to stop the goroutine
	done := make(chan struct{})

	go func() {
		for {
			_, _, err := ws.ReadMessage()
			if err != nil {
				close(done)
				return
			}
		}
	}()

	select {
	case <-done:
		logUI.Log.Debug("Connection closed by client")
		return nil
	case <-wsManager.broadcastStop:
		return nil
	}
}

func (wsManager *WebSocketManager) broadcastEngineState() {
	defer func() {
		if r := recover(); r != nil {
			fmt.Println("Recovered from panic:", r)
		}
	}()
	wsManager.engineState.Update()
	msg, err := msgpack.Marshal(wsManager.engineState)
	if err != nil {
		logUI.Log.Err("Error marshaling combined message: %v", err)
		return
	}
	wsManager.connections.broadcast(msg)
	// Reset the refresh flag
	wsManager.engineState.Preview.Refresh = false
}

func (wsManager *WebSocketManager) startBroadcastLoop() {
	wsManager.broadcastStart.Do(func() {
		go func() {
			for {
				select {
				case <-wsManager.broadcastStop:
					return
				default:
					if engine.NSteps != wsManager.lastStep {
						if len(wsManager.connections.conns) > 0 {
							wsManager.broadcastEngineState()
							wsManager.lastStep = engine.NSteps
						}
					}
					time.Sleep(1 * time.Second)
				}
			}
		}()
	})
}
