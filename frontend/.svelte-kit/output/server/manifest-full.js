export const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set(["favicon.ico","prism.css","favicon-32x32.png","apple-touch-icon.png","favicon-16x16.png"]),
	mimeTypes: {".css":"text/css",".png":"image/png"},
	_: {
		client: {"start":"_app/immutable/entry/start.DVI6hWkl.js","app":"_app/immutable/entry/app.CFNeIeHv.js","imports":["_app/immutable/entry/start.DVI6hWkl.js","_app/immutable/chunks/entry.NmHHfd6V.js","_app/immutable/chunks/utils.CzYqvc2b.js","_app/immutable/chunks/index.sNV_l9ib.js","_app/immutable/entry/app.CFNeIeHv.js","_app/immutable/chunks/utils.CzYqvc2b.js","_app/immutable/chunks/disclose-version.DDWHpLfT.js","_app/immutable/chunks/index-client.CLrowoUG.js","_app/immutable/chunks/this.DaNKizv3.js"],"stylesheets":[],"fonts":[],"uses_env_dynamic_public":false},
		nodes: [
			__memo(() => import('./nodes/0.js')),
			__memo(() => import('./nodes/1.js')),
			__memo(() => import('./nodes/2.js'))
		],
		routes: [
			{
				id: "/",
				pattern: /^\/$/,
				params: [],
				page: { layouts: [0,], errors: [1,], leaf: 2 },
				endpoint: null
			}
		],
		matchers: async () => {
			
			return {  };
		},
		server_assets: {}
	}
}
})();
