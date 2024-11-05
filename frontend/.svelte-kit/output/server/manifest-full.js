export const manifest = (() => {
function __memo(fn) {
	let value;
	return () => value ??= (value = fn());
}

return {
	appDir: "_app",
	appPath: "_app",
	assets: new Set(["favicon-16x16.png","prism.css","favicon.ico","favicon-32x32.png","apple-touch-icon.png","favicon.png"]),
	mimeTypes: {".png":"image/png",".css":"text/css"},
	_: {
		client: {"start":"_app/immutable/entry/start.CQ93jRdN.js","app":"_app/immutable/entry/app.DjNWUGnc.js","imports":["_app/immutable/entry/start.CQ93jRdN.js","_app/immutable/chunks/entry.DivLhnXe.js","_app/immutable/chunks/runtime.CW_FmKor.js","_app/immutable/chunks/index.DvAu-9B3.js","_app/immutable/entry/app.DjNWUGnc.js","_app/immutable/chunks/runtime.CW_FmKor.js","_app/immutable/chunks/disclose-version.CbuMJ-8b.js","_app/immutable/chunks/index-client.BGcNfKnx.js","_app/immutable/chunks/this.fQtdachP.js"],"stylesheets":[],"fonts":[],"uses_env_dynamic_public":false},
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
