import { j as rest_props, k as fallback, t as getContext, l as spread_attributes, u as escape_html, n as slot, o as bind_props, f as pop, q as sanitize_props, p as push, y as spread_props, F as invalid_default_snippet, m as attr } from "./index2.js";
import { twMerge } from "tailwind-merge";
import { w as writable } from "./index.js";
function ToolbarButton($$payload, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const $$restProps = rest_props($$sanitized_props, [
    "color",
    "name",
    "ariaLabel",
    "size",
    "href"
  ]);
  push();
  let color = fallback($$props["color"], "default");
  let name = fallback($$props["name"], () => void 0, true);
  let ariaLabel = fallback($$props["ariaLabel"], () => void 0, true);
  let size = fallback($$props["size"], "md");
  let href = fallback($$props["href"], () => void 0, true);
  const background = getContext("background");
  const colors = {
    dark: "text-gray-500 hover:text-gray-900 hover:bg-gray-200 dark:text-gray-400 dark:hover:text-white dark:hover:bg-gray-600",
    gray: "text-gray-500 focus:ring-gray-400 hover:bg-gray-200 dark:hover:bg-gray-800 dark:hover:text-gray-300",
    red: "text-red-500 focus:ring-red-400 hover:bg-red-200 dark:hover:bg-red-800 dark:hover:text-red-300",
    yellow: "text-yellow-500 focus:ring-yellow-400 hover:bg-yellow-200 dark:hover:bg-yellow-800 dark:hover:text-yellow-300",
    green: "text-green-500 focus:ring-green-400 hover:bg-green-200 dark:hover:bg-green-800 dark:hover:text-green-300",
    indigo: "text-indigo-500 focus:ring-indigo-400 hover:bg-indigo-200 dark:hover:bg-indigo-800 dark:hover:text-indigo-300",
    purple: "text-purple-500 focus:ring-purple-400 hover:bg-purple-200 dark:hover:bg-purple-800 dark:hover:text-purple-300",
    pink: "text-pink-500 focus:ring-pink-400 hover:bg-pink-200 dark:hover:bg-pink-800 dark:hover:text-pink-300",
    blue: "text-blue-500 focus:ring-blue-400 hover:bg-blue-200 dark:hover:bg-blue-800 dark:hover:text-blue-300",
    primary: "text-primary-500 focus:ring-primary-400 hover:bg-primary-200 dark:hover:bg-primary-800 dark:hover:text-primary-300",
    default: "focus:ring-gray-400 hover:bg-gray-100"
  };
  const sizing = {
    xs: "m-0.5 rounded-sm focus:ring-1 p-0.5",
    sm: "m-0.5 rounded focus:ring-1 p-0.5",
    md: "m-0.5 rounded-lg focus:ring-2 p-1.5",
    lg: "m-0.5 rounded-lg focus:ring-2 p-2.5"
  };
  let buttonClass;
  const svgSizes = {
    xs: "w-3 h-3",
    sm: "w-3.5 h-3.5",
    md: "w-5 h-5",
    lg: "w-5 h-5"
  };
  buttonClass = twMerge("focus:outline-none whitespace-normal", sizing[size], colors[color], color === "default" && (background ? "dark:hover:bg-gray-600" : "dark:hover:bg-gray-700"), $$sanitized_props.class);
  if (href) {
    $$payload.out += "<!--[-->";
    $$payload.out += `<a${spread_attributes({
      href,
      ...$$restProps,
      class: buttonClass,
      "aria-label": ariaLabel ?? name
    })}>`;
    if (name) {
      $$payload.out += "<!--[-->";
      $$payload.out += `<span class="sr-only">${escape_html(name)}</span>`;
    } else {
      $$payload.out += "<!--[!-->";
    }
    $$payload.out += `<!--]--> <!---->`;
    slot($$payload, $$props, "default", { svgSize: svgSizes[size] }, null);
    $$payload.out += `<!----></a>`;
  } else {
    $$payload.out += "<!--[!-->";
    $$payload.out += `<button${spread_attributes({
      type: "button",
      ...$$restProps,
      class: buttonClass,
      "aria-label": ariaLabel ?? name
    })}>`;
    if (name) {
      $$payload.out += "<!--[-->";
      $$payload.out += `<span class="sr-only">${escape_html(name)}</span>`;
    } else {
      $$payload.out += "<!--[!-->";
    }
    $$payload.out += `<!--]--> <!---->`;
    slot($$payload, $$props, "default", { svgSize: svgSizes[size] }, null);
    $$payload.out += `<!----></button>`;
  }
  $$payload.out += `<!--]-->`;
  bind_props($$props, { color, name, ariaLabel, size, href });
  pop();
}
function CloseButton($$payload, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const $$restProps = rest_props($$sanitized_props, ["name"]);
  push();
  let name = fallback($$props["name"], "Close");
  ToolbarButton($$payload, spread_props([
    { name },
    $$restProps,
    {
      class: twMerge("ms-auto", $$sanitized_props.class),
      children: invalid_default_snippet,
      $$slots: {
        default: ($$payload2, { svgSize }) => {
          $$payload2.out += `<svg${attr("class", svgSize)} fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path></svg>`;
        }
      }
    }
  ]));
  bind_props($$props, { name });
  pop();
}
let alert = writable("");
function setAlert(message) {
  alert.set(message);
  setTimeout(() => {
    alert.set("");
  }, 1e4);
}
export {
  CloseButton as C,
  alert as a,
  setAlert as s
};
