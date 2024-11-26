import { T as sanitize_slots, V as rest_props, W as fallback, X as spread_attributes, Y as attr, Z as slot, _ as bind_props, S as pop, $ as sanitize_props, Q as push, a0 as getContext, a1 as escape_html, a2 as store_get, a3 as unsubscribe_stores } from "../../chunks/index.js";
import { twMerge } from "tailwind-merge";
import { C as CloseButton, a as alert } from "../../chunks/alert.js";
import "msgpack-lite";
import "echarts";
const linear = (x) => x;
function fade(node, { delay = 0, duration = 400, easing = linear } = {}) {
  const o = +getComputedStyle(node).opacity;
  return {
    delay,
    duration,
    easing,
    css: (t) => `opacity: ${t * o}`
  };
}
function Toast($$payload, $$props) {
  const $$slots = sanitize_slots($$props);
  const $$sanitized_props = sanitize_props($$props);
  const $$restProps = rest_props($$sanitized_props, [
    "dismissable",
    "color",
    "position",
    "divClass",
    "defaultIconClass",
    "contentClass",
    "align",
    "transition",
    "params",
    "toastStatus"
  ]);
  push();
  let dismissable = fallback($$props["dismissable"], true);
  let color = fallback($$props["color"], "primary");
  let position = fallback($$props["position"], "none");
  let divClass = fallback($$props["divClass"], "w-full max-w-xs p-4 text-gray-500 bg-white shadow dark:text-gray-400 dark:bg-gray-800 gap-3");
  let defaultIconClass = fallback($$props["defaultIconClass"], "w-8 h-8");
  let contentClass = fallback($$props["contentClass"], "w-full text-sm font-normal");
  let align = fallback($$props["align"], true);
  let transition = fallback($$props["transition"], fade);
  let params = fallback($$props["params"], () => ({}), true);
  let toastStatus = fallback($$props["toastStatus"], true);
  const positions = {
    "top-left": "absolute top-5 start-5",
    "top-right": "absolute top-5 end-5",
    "bottom-left": "absolute bottom-5 start-5",
    "bottom-right": "absolute bottom-5 end-5",
    none: ""
  };
  let finalDivClass = twMerge("flex", align ? "items-center" : "items-start", divClass, positions[position], $$sanitized_props.class);
  const colors = {
    primary: "text-primary-500 bg-primary-100 dark:bg-primary-800 dark:text-primary-200",
    dark: "text-gray-500 bg-gray-100 dark:bg-gray-700 dark:text-gray-200",
    red: "text-red-500 bg-red-100 dark:bg-red-800 dark:text-red-200",
    yellow: "text-yellow-500 bg-yellow-100 dark:bg-yellow-800 dark:text-yellow-200",
    green: "text-green-500 bg-green-100 dark:bg-green-800 dark:text-green-200",
    blue: "text-blue-500 bg-blue-100 dark:bg-blue-800 dark:text-blue-200",
    indigo: "text-indigo-500 bg-indigo-100 dark:bg-indigo-800 dark:text-indigo-200",
    purple: "text-purple-500 bg-purple-100 dark:bg-purple-800 dark:text-purple-200",
    pink: "text-pink-500 bg-pink-100 dark:bg-pink-700 dark:text-pink-200",
    none: ""
  };
  let iconClass;
  const clsBtnExtraClass = "ms-auto -mx-1.5 -my-1.5 bg-white text-gray-400 hover:text-gray-900 rounded-lg focus:ring-2 focus:ring-gray-300 p-1.5 hover:bg-gray-100 inline-flex items-center justify-center h-8 w-8 dark:text-gray-500 dark:hover:text-white dark:bg-gray-800 dark:hover:bg-gray-700";
  iconClass = twMerge("inline-flex items-center justify-center shrink-0 rounded-lg", colors[color], defaultIconClass);
  if (toastStatus) {
    $$payload.out += "<!--[-->";
    $$payload.out += `<div${spread_attributes({
      role: "alert",
      ...$$restProps,
      class: finalDivClass
    })}>`;
    if ($$slots.icon) {
      $$payload.out += "<!--[-->";
      $$payload.out += `<div${attr("class", iconClass)}><!---->`;
      slot($$payload, $$props, "icon", {}, null);
      $$payload.out += `<!----></div>`;
    } else {
      $$payload.out += "<!--[!-->";
    }
    $$payload.out += `<!--]--> <div${attr("class", contentClass)}><!---->`;
    slot($$payload, $$props, "default", {}, null);
    $$payload.out += `<!----></div> `;
    if (dismissable) {
      $$payload.out += "<!--[-->";
      CloseButton($$payload, {
        divclass: clsBtnExtraClass,
        ariaLabel: "Remove toast",
        color
      });
    } else {
      $$payload.out += "<!--[!-->";
    }
    $$payload.out += `<!--]--></div>`;
  } else {
    $$payload.out += "<!--[!-->";
  }
  $$payload.out += `<!--]-->`;
  bind_props($$props, {
    dismissable,
    color,
    position,
    divClass,
    defaultIconClass,
    contentClass,
    align,
    transition,
    params,
    toastStatus
  });
  pop();
}
function ExclamationCircleSolid($$payload, $$props) {
  push();
  const ctx = getContext("iconCtx") ?? {};
  const sizes = {
    xs: "w-3 h-3",
    sm: "w-4 h-4",
    md: "w-5 h-5",
    lg: "w-6 h-6",
    xl: "w-8 h-8"
  };
  let {
    size = ctx.size || "md",
    color = ctx.color || "currentColor",
    title,
    desc,
    class: className,
    ariaLabel = "exclamation circle solid",
    $$slots,
    $$events,
    ...restProps
  } = $$props;
  let ariaDescribedby = `${title?.id || ""} ${desc?.id || ""}`;
  const hasDescription = !!(title?.id || desc?.id);
  $$payload.out += `<svg${spread_attributes(
    {
      xmlns: "http://www.w3.org/2000/svg",
      fill: color,
      ...restProps,
      class: twMerge("shrink-0", sizes[size], className),
      "aria-label": ariaLabel,
      "aria-describedby": hasDescription ? ariaDescribedby : void 0,
      viewBox: "0 0 24 24"
    },
    void 0,
    void 0,
    3
  )}>`;
  if (title?.id && title.title) {
    $$payload.out += "<!--[-->";
    $$payload.out += `<title${attr("id", title.id)}>${escape_html(title.title)}</title>`;
  } else {
    $$payload.out += "<!--[!-->";
  }
  $$payload.out += `<!--]-->`;
  if (desc?.id && desc.desc) {
    $$payload.out += "<!--[-->";
    $$payload.out += `<desc${attr("id", desc.id)}>${escape_html(desc.desc)}</desc>`;
  } else {
    $$payload.out += "<!--[!-->";
  }
  $$payload.out += `<!--]--><path fill-rule="evenodd" d="M2 12C2 6.477 6.477 2 12 2s10 4.477 10 10-4.477 10-10 10S2 17.523 2 12Zm11-4a1 1 0 1 0-2 0v5a1 1 0 1 0 2 0V8Zm-1 7a1 1 0 1 0 0 2h.01a1 1 0 1 0 0-2H12Z" clip-rule="evenodd"></path></svg>`;
  pop();
}
function Alert($$payload) {
  var $$store_subs;
  if (store_get($$store_subs ??= {}, "$alert", alert) !== "") {
    $$payload.out += "<!--[-->";
    $$payload.out += `<div class="fixed left-3/4 top-10 z-50 gap-10">`;
    Toast($$payload, {
      color: "red",
      children: ($$payload2) => {
        $$payload2.out += `<!---->${escape_html(store_get($$store_subs ??= {}, "$alert", alert))}`;
      },
      $$slots: {
        default: true,
        icon: ($$payload2) => {
          {
            ExclamationCircleSolid($$payload2, { class: "h-5 w-5" });
            $$payload2.out += `<!----> <span class="sr-only">Warning icon</span>`;
          }
        }
      }
    });
    $$payload.out += `<!----></div>`;
  } else {
    $$payload.out += "<!--[!-->";
  }
  $$payload.out += `<!--]-->`;
  if ($$store_subs) unsubscribe_stores($$store_subs);
}
function _layout($$payload, $$props) {
  push();
  Alert($$payload);
  $$payload.out += `<!----> <!---->`;
  slot($$payload, $$props, "default", {}, null);
  $$payload.out += `<!---->`;
  pop();
}
export {
  _layout as default
};
