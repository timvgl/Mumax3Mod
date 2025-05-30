
// this file is generated — do not edit it


/// <reference types="@sveltejs/kit" />

/**
 * Environment variables [loaded by Vite](https://vitejs.dev/guide/env-and-mode.html#env-files) from `.env` files and `process.env`. Like [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), this module cannot be imported into client-side code. This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured).
 * 
 * _Unlike_ [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), the values exported from this module are statically injected into your bundle at build time, enabling optimisations like dead code elimination.
 * 
 * ```ts
 * import { API_KEY } from '$env/static/private';
 * ```
 * 
 * Note that all environment variables referenced in your code should be declared (for example in an `.env` file), even if they don't have a value until the app is deployed:
 * 
 * ```
 * MY_FEATURE_FLAG=""
 * ```
 * 
 * You can override `.env` values from the command line like so:
 * 
 * ```bash
 * MY_FEATURE_FLAG="enabled" npm run dev
 * ```
 */
declare module '$env/static/private' {
	export const SHELL: string;
	export const npm_command: string;
	export const SESSION_MANAGER: string;
	export const QT_ACCESSIBILITY: string;
	export const COLORTERM: string;
	export const WSL2_GUI_APPS_ENABLED: string;
	export const TERM_PROGRAM_VERSION: string;
	export const WSL_DISTRO_NAME: string;
	export const NODE: string;
	export const CGO_LDFLAGS: string;
	export const npm_config_local_prefix: string;
	export const NAME: string;
	export const PWD: string;
	export const LOGNAME: string;
	export const _: string;
	export const XAUTHORITY: string;
	export const VSCODE_GIT_ASKPASS_NODE: string;
	export const HOME: string;
	export const LANG: string;
	export const WSL_INTEROP: string;
	export const LS_COLORS: string;
	export const XDG_CURRENT_DESKTOP: string;
	export const npm_package_version: string;
	export const WAYLAND_DISPLAY: string;
	export const GIT_ASKPASS: string;
	export const VSCODE_GIT_ASKPASS_EXTRA_ARGS: string;
	export const TERM: string;
	export const npm_package_name: string;
	export const CPLUS_INCLUDE_PATH: string;
	export const USER: string;
	export const VSCODE_GIT_IPC_HANDLE: string;
	export const DISPLAY: string;
	export const npm_lifecycle_event: string;
	export const SHLVL: string;
	export const npm_config_user_agent: string;
	export const npm_execpath: string;
	export const XDG_RUNTIME_DIR: string;
	export const npm_package_json: string;
	export const WSLENV: string;
	export const VSCODE_GIT_ASKPASS_MAIN: string;
	export const CUDA_HOME: string;
	export const PATH: string;
	export const C_INCLUDE_PATH: string;
	export const HOSTTYPE: string;
	export const PULSE_SERVER: string;
	export const npm_node_execpath: string;
	export const OLDPWD: string;
	export const GOPATH: string;
	export const TERM_PROGRAM: string;
	export const NODE_ENV: string;
}

/**
 * Similar to [`$env/static/private`](https://svelte.dev/docs/kit/$env-static-private), except that it only includes environment variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`), and can therefore safely be exposed to client-side code.
 * 
 * Values are replaced statically at build time.
 * 
 * ```ts
 * import { PUBLIC_BASE_URL } from '$env/static/public';
 * ```
 */
declare module '$env/static/public' {
	
}

/**
 * This module provides access to runtime environment variables, as defined by the platform you're running on. For example if you're using [`adapter-node`](https://github.com/sveltejs/kit/tree/main/packages/adapter-node) (or running [`vite preview`](https://svelte.dev/docs/kit/cli)), this is equivalent to `process.env`. This module only includes variables that _do not_ begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) _and do_ start with [`config.kit.env.privatePrefix`](https://svelte.dev/docs/kit/configuration#env) (if configured).
 * 
 * This module cannot be imported into client-side code.
 * 
 * Dynamic environment variables cannot be used during prerendering.
 * 
 * ```ts
 * import { env } from '$env/dynamic/private';
 * console.log(env.DEPLOYMENT_SPECIFIC_VARIABLE);
 * ```
 * 
 * > In `dev`, `$env/dynamic` always includes environment variables from `.env`. In `prod`, this behavior will depend on your adapter.
 */
declare module '$env/dynamic/private' {
	export const env: {
		SHELL: string;
		npm_command: string;
		SESSION_MANAGER: string;
		QT_ACCESSIBILITY: string;
		COLORTERM: string;
		WSL2_GUI_APPS_ENABLED: string;
		TERM_PROGRAM_VERSION: string;
		WSL_DISTRO_NAME: string;
		NODE: string;
		CGO_LDFLAGS: string;
		npm_config_local_prefix: string;
		NAME: string;
		PWD: string;
		LOGNAME: string;
		_: string;
		XAUTHORITY: string;
		VSCODE_GIT_ASKPASS_NODE: string;
		HOME: string;
		LANG: string;
		WSL_INTEROP: string;
		LS_COLORS: string;
		XDG_CURRENT_DESKTOP: string;
		npm_package_version: string;
		WAYLAND_DISPLAY: string;
		GIT_ASKPASS: string;
		VSCODE_GIT_ASKPASS_EXTRA_ARGS: string;
		TERM: string;
		npm_package_name: string;
		CPLUS_INCLUDE_PATH: string;
		USER: string;
		VSCODE_GIT_IPC_HANDLE: string;
		DISPLAY: string;
		npm_lifecycle_event: string;
		SHLVL: string;
		npm_config_user_agent: string;
		npm_execpath: string;
		XDG_RUNTIME_DIR: string;
		npm_package_json: string;
		WSLENV: string;
		VSCODE_GIT_ASKPASS_MAIN: string;
		CUDA_HOME: string;
		PATH: string;
		C_INCLUDE_PATH: string;
		HOSTTYPE: string;
		PULSE_SERVER: string;
		npm_node_execpath: string;
		OLDPWD: string;
		GOPATH: string;
		TERM_PROGRAM: string;
		NODE_ENV: string;
		[key: `PUBLIC_${string}`]: undefined;
		[key: `${string}`]: string | undefined;
	}
}

/**
 * Similar to [`$env/dynamic/private`](https://svelte.dev/docs/kit/$env-dynamic-private), but only includes variables that begin with [`config.kit.env.publicPrefix`](https://svelte.dev/docs/kit/configuration#env) (which defaults to `PUBLIC_`), and can therefore safely be exposed to client-side code.
 * 
 * Note that public dynamic environment variables must all be sent from the server to the client, causing larger network requests — when possible, use `$env/static/public` instead.
 * 
 * Dynamic environment variables cannot be used during prerendering.
 * 
 * ```ts
 * import { env } from '$env/dynamic/public';
 * console.log(env.PUBLIC_DEPLOYMENT_SPECIFIC_VARIABLE);
 * ```
 */
declare module '$env/dynamic/public' {
	export const env: {
		[key: `PUBLIC_${string}`]: string | undefined;
	}
}
