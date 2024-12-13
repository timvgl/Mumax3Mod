package api

import "github.com/labstack/echo/v4"

type EngineState struct {
	Header    *HeaderState     `msgpack:"header"`
	Console   *ConsoleState    `msgpack:"console"`
	Preview   *PreviewState    `msgpack:"preview"`
	Solver    *SolverState     `msgpack:"solver"`
	Mesh      *MeshState       `msgpack:"mesh"`
	Params    *ParametersState `msgpack:"parameters"`
	TablePlot *TablePlotState  `msgpack:"tablePlot"`
	Metrics   *MetricsState    `msgpack:"metrics"`
}

func initEngineStateAPI(e *echo.Group, ws *WebSocketManager) *EngineState {
	preview := initPreviewAPI(e, ws)
	return &EngineState{
		Header:    initHeaderAPI(),
		Console:   initConsoleAPI(e, ws),
		Preview:   preview,
		Solver:    initSolverAPI(e, ws),
		Mesh:      initMeshAPI(e, ws, *preview),
		Params:    initParameterAPI(e, ws),
		TablePlot: initTablePlotAPI(e, ws),
		Metrics:   initMetricsAPI(e, ws),
	}
}

func (es *EngineState) Update() {
	es.Header.Update()
	es.Console.Update()
	es.Preview.Update()
	es.Solver.Update()
	es.Mesh.Update(*es.Preview)
	es.Params.Update()
	es.TablePlot.Update()
	es.Metrics.Update()
}
