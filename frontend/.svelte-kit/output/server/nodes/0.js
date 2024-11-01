import * as universal from '../entries/pages/_layout.js';

export const index = 0;
let component_cache;
export const component = async () => component_cache ??= (await import('../entries/pages/_layout.svelte.js')).default;
export { universal };
export const universal_id = "src/routes/+layout.js";
export const imports = ["_app/immutable/nodes/0.BaZAfoDr.js","_app/immutable/chunks/disclose-version.CbuMJ-8b.js","_app/immutable/chunks/runtime.CW_FmKor.js","_app/immutable/chunks/websocket.Dvjyuv75.js","_app/immutable/chunks/store.C89PaXyH.js","_app/immutable/chunks/index.DvAu-9B3.js","_app/immutable/chunks/index-client.BGcNfKnx.js"];
export const stylesheets = ["_app/immutable/assets/0.CD-nSkxO.css"];
export const fonts = [];
