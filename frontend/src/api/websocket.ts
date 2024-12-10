import msgpack from 'msgpack-lite';

import { type Preview, previewState } from './incoming/preview';
import { type Header, headerState } from './incoming/header';
import { type Solver, solverState } from './incoming/solver';
import { type Console, consoleState } from './incoming/console';
import { type Mesh, meshState } from './incoming/mesh';
import { type Parameters, parametersState, sortFieldsByName } from './incoming/parameters';
import { type TablePlot, tablePlotState } from './incoming/table-plot';
import { get, writable } from 'svelte/store';
import { preview3D } from '$lib/preview/preview3D';
import { preview2D } from '$lib/preview/preview2D';
import { plotTable } from '$lib/table-plot/table-plot';
import { metricsState } from './incoming/metrics';

export let connected = writable(false);

export function initializeWebSocket() {
	let retryInterval = 1000;
	let ws: WebSocket | null = null;

	function connect() {
		let wsUrl = './ws';
		console.debug('Connecting to WebSocket server at', wsUrl);
		ws = new WebSocket(wsUrl);
		ws.binaryType = 'arraybuffer';

		ws.onopen = function () {
			console.debug('WebSocket connection established');
			connected.set(true);
			// Clear the connection timeout if it was set in tryConnect
			if (connectionTimeout) {
				clearTimeout(connectionTimeout);
				connectionTimeout = null;
			}
		};

		ws.onmessage = function (event) {
			parseMsgpack(event.data);
			ws?.send('ok');
			connected.set(true);
		};

		ws.onclose = function () {
			connected.set(false);
			console.debug(
				'WebSocket closed. Attempting to reconnect in ' + retryInterval / 1000 + ' seconds...'
			);
			ws = null; // Ensure ws is set to null when closed
			// Clear the connection timeout if it was set in tryConnect
			if (connectionTimeout) {
				clearTimeout(connectionTimeout);
				connectionTimeout = null;
			}
			setTimeout(tryConnect, retryInterval);
		};

		ws.onerror = function (event) {
			console.error('WebSocket encountered error:', event);
			// Clear the connection timeout if it was set in tryConnect
			if (connectionTimeout) {
				clearTimeout(connectionTimeout);
				connectionTimeout = null;
			}
			if (ws) {
				ws.close();
			}
		};
	}

	let connectionTimeout: number | null = null;

	function tryConnect() {
		console.debug('Attempting WebSocket connection...');
		try {
			connect();

			// If the connection does not open within retryInterval, consider it a timeout
			connectionTimeout = window.setTimeout(() => {
				console.error(`WebSocket connection timed out after ${3*retryInterval}ms`);
				if (ws && ws.readyState !== WebSocket.OPEN) {
					ws.close();
					ws = null;
					setTimeout(tryConnect, 3*retryInterval);
				}
			}, 3*retryInterval);
		} catch (err) {
			console.error(
				'WebSocket connection failed:',
				err,
				'Retrying in ' + retryInterval / 1000 + ' seconds...'
			);
			if (connectionTimeout) {
				clearTimeout(connectionTimeout);
				connectionTimeout = null;
			}
			setTimeout(tryConnect, retryInterval);
		}
	}

	tryConnect();
}

export function parseMsgpack(data: ArrayBuffer) {
	const msg = msgpack.decode(new Uint8Array(data));
	consoleState.set(msg.console as Console);

	headerState.set(msg.header as Header);

	meshState.set(msg.mesh as Mesh);

	parametersState.set(msg.parameters as Parameters);
	sortFieldsByName();

	solverState.set(msg.solver as Solver);

	tablePlotState.set(msg.tablePlot as TablePlot);
	plotTable();

	previewState.set(msg.preview as Preview);
	if (get(previewState).type === '3D') {
		preview3D();
	} else {
		preview2D();
	}

	metricsState.set(msg.metrics);
}
