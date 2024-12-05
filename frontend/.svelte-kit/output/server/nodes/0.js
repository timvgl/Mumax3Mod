import * as universal from '../entries/pages/_layout.js';

export const index = 0;
let component_cache;
export const component = async () => component_cache ??= (await import('../entries/pages/_layout.svelte.js')).default;
export { universal };
export const universal_id = "src/routes/+layout.js";
export const imports = ["_app/immutable/nodes/0.syKpzFVd.js","_app/immutable/chunks/disclose-version.DDWHpLfT.js","_app/immutable/chunks/utils.CzYqvc2b.js","_app/immutable/chunks/legacy.BD-xdWD-.js","_app/immutable/chunks/websocket.D6x40JCv.js","_app/immutable/chunks/index-client.CLrowoUG.js","_app/immutable/chunks/index.sNV_l9ib.js"];
export const stylesheets = ["_app/immutable/assets/0.D-gji24D.css"];
export const fonts = [];
