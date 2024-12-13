package api

import (
	"maps"
	"net/http"
	"slices"

	"github.com/labstack/echo/v4"
	"github.com/mumax/3/engine"
)

var oldQMesh engine.Quantity

type MeshState struct {
	ws   *WebSocketManager
	Dx   float64 `msgpack:"dx"`
	Dy   float64 `msgpack:"dy"`
	Dz   float64 `msgpack:"dz"`
	Nx   int     `msgpack:"Nx"`
	Ny   int     `msgpack:"Ny"`
	Nz   int     `msgpack:"Nz"`
	Tx   float64 `msgpack:"Tx"`
	Ty   float64 `msgpack:"Ty"`
	Tz   float64 `msgpack:"Tz"`
	PBCx int     `msgpack:"PBCx"`
	PBCy int     `msgpack:"PBCy"`
	PBCz int     `msgpack:"PBCz"`
}

func initMeshAPI(e *echo.Group, ws *WebSocketManager, Preview PreviewState) *MeshState {
	meshState := MeshState{
		ws:   ws,
		Dx:   engine.Dx,
		Dy:   engine.Dy,
		Dz:   engine.Dz,
		Nx:   engine.Nx,
		Ny:   engine.Ny,
		Nz:   engine.Nz,
		Tx:   engine.Tx,
		Ty:   engine.Ty,
		Tz:   engine.Tz,
		PBCx: engine.PBCx,
		PBCy: engine.PBCy,
		PBCz: engine.PBCz,
	}
	e.POST("/api/mesh", meshState.postMesh)
	oldQMesh = Preview.getQuantity()
	return &meshState
}

func (m *MeshState) Update(Preview PreviewState) {
	if slices.Contains(slices.Concat(slices.Collect(maps.Values(Preview.DynQuantities))...), Preview.Quantity) && oldQMesh != Preview.getQuantity() {
		mesh := engine.MeshOf(Preview.getQuantity())
		oldQMesh = Preview.getQuantity()
		m.Nx = mesh.Nx()
		m.Ny = mesh.Ny()
		m.Nz = mesh.Nz()
		dxdydz := mesh.CellSize()
		m.Dx = dxdydz[0]
		m.Dy = dxdydz[1]
		m.Dz = dxdydz[2]
		m.Tx = float64(m.Nx) * m.Dx
		m.Ty = float64(m.Ny) * m.Dy
		m.Tz = float64(m.Nz) * m.Dz

	} else if !slices.Contains(slices.Concat(slices.Collect(maps.Values(Preview.DynQuantities))...), Preview.Quantity) && oldQMesh != Preview.getQuantity() {
		oldQMesh = Preview.getQuantity()
		m.Nx = engine.Nx
		m.Ny = engine.Ny
		m.Nz = engine.Nz
		m.Dx = engine.Dx
		m.Dy = engine.Dy
		m.Dz = engine.Dz
		m.Tx = engine.Tx
		m.Ty = engine.Ty
		m.Tz = engine.Tz
	}
}

func (m *MeshState) postMesh(c echo.Context) error {
	return c.JSON(http.StatusNotImplemented, "")
}
