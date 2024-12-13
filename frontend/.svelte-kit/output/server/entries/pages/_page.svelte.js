import { a4 as noop, V as rest_props, R as setContext, W as fallback, a5 as element, _ as bind_props, S as pop, $ as sanitize_props, Q as push, Z as slot, X as spread_attributes, a0 as getContext, a6 as spread_props, Y as attr, a7 as copy_payload, a8 as assign_payload, T as sanitize_slots, a9 as ensure_array_like, a1 as escape_html, aa as stringify, a2 as store_get, a3 as unsubscribe_stores, ab as store_mutate } from "../../chunks/index.js";
import { w as writable } from "../../chunks/index2.js";
import "msgpack-lite";
import "echarts";
import { twMerge, twJoin } from "tailwind-merge";
import * as dom from "@floating-ui/dom";
import { C as CloseButton, s as setAlert } from "../../chunks/alert.js";
import Prism from "prismjs";
import "prismjs/components/prism-go.js";
const now = () => Date.now();
const raf = {
  // don't access requestAnimationFrame eagerly outside method
  // this allows basic testing of user code without JSDOM
  // bunder will eval and remove ternary when the user's app is built
  tick: (
    /** @param {any} _ */
    (_) => noop()
  ),
  now: () => now(),
  tasks: /* @__PURE__ */ new Set()
};
function loop(callback) {
  let task;
  if (raf.tasks.size === 0) ;
  return {
    promise: new Promise((fulfill) => {
      raf.tasks.add(task = { c: callback, f: fulfill });
    }),
    abort() {
      raf.tasks.delete(task);
    }
  };
}
function html(value) {
  var html2 = String(value ?? "");
  var open = "<!---->";
  return open + html2 + "<!---->";
}
const bgColors = {
  gray: "bg-gray-50 dark:bg-gray-800",
  red: "bg-red-50 dark:bg-gray-800",
  yellow: "bg-yellow-50 dark:bg-gray-800 ",
  green: "bg-green-50 dark:bg-gray-800 ",
  indigo: "bg-indigo-50 dark:bg-gray-800 ",
  purple: "bg-purple-50 dark:bg-gray-800 ",
  pink: "bg-pink-50 dark:bg-gray-800 ",
  blue: "bg-blue-50 dark:bg-gray-800 ",
  light: "bg-gray-50 dark:bg-gray-700",
  dark: "bg-gray-50 dark:bg-gray-800",
  default: "bg-white dark:bg-gray-800",
  dropdown: "bg-white dark:bg-gray-700",
  navbar: "bg-white dark:bg-gray-900",
  navbarUl: "bg-gray-50 dark:bg-gray-800",
  form: "bg-gray-50 dark:bg-gray-700",
  primary: "bg-primary-50 dark:bg-gray-800 ",
  orange: "bg-orange-50 dark:bg-orange-800",
  none: ""
};
function Frame($$payload, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const $$restProps = rest_props($$sanitized_props, [
    "tag",
    "color",
    "rounded",
    "border",
    "shadow",
    "node",
    "use",
    "options",
    "role",
    "transition",
    "params",
    "open"
  ]);
  push();
  const noop2 = () => {
  };
  setContext("background", true);
  let tag = fallback($$props["tag"], () => $$restProps.href ? "a" : "div", true);
  let color = fallback($$props["color"], "default");
  let rounded = fallback($$props["rounded"], false);
  let border = fallback($$props["border"], false);
  let shadow = fallback($$props["shadow"], false);
  let node = fallback($$props["node"], () => void 0, true);
  let use = fallback($$props["use"], noop2);
  let options = fallback($$props["options"], () => ({}), true);
  let role = fallback($$props["role"], () => void 0, true);
  let transition = fallback($$props["transition"], () => void 0, true);
  let params = fallback($$props["params"], () => ({}), true);
  let open = fallback($$props["open"], true);
  const textColors = {
    gray: "text-gray-800 dark:text-gray-300",
    red: "text-red-800 dark:text-red-400",
    yellow: "text-yellow-800 dark:text-yellow-300",
    green: "text-green-800 dark:text-green-400",
    indigo: "text-indigo-800 dark:text-indigo-400",
    purple: "text-purple-800 dark:text-purple-400",
    pink: "text-pink-800 dark:text-pink-400",
    blue: "text-blue-800 dark:text-blue-400",
    light: "text-gray-700 dark:text-gray-300",
    dark: "text-gray-700 dark:text-gray-300",
    default: "text-gray-500 dark:text-gray-400",
    dropdown: "text-gray-700 dark:text-gray-200",
    navbar: "text-gray-700 dark:text-gray-200",
    navbarUl: "text-gray-700 dark:text-gray-400",
    form: "text-gray-900 dark:text-white",
    primary: "text-primary-800 dark:text-primary-400",
    orange: "text-orange-800 dark:text-orange-400",
    none: ""
  };
  const borderColors = {
    gray: "border-gray-300 dark:border-gray-800 divide-gray-300 dark:divide-gray-800",
    red: "border-red-300 dark:border-red-800 divide-red-300 dark:divide-red-800",
    yellow: "border-yellow-300 dark:border-yellow-800 divide-yellow-300 dark:divide-yellow-800",
    green: "border-green-300 dark:border-green-800 divide-green-300 dark:divide-green-800",
    indigo: "border-indigo-300 dark:border-indigo-800 divide-indigo-300 dark:divide-indigo-800",
    purple: "border-purple-300 dark:border-purple-800 divide-purple-300 dark:divide-purple-800",
    pink: "border-pink-300 dark:border-pink-800 divide-pink-300 dark:divide-pink-800",
    blue: "border-blue-300 dark:border-blue-800 divide-blue-300 dark:divide-blue-800",
    light: "border-gray-500 divide-gray-500",
    dark: "border-gray-500 divide-gray-500",
    default: "border-gray-200 dark:border-gray-700 divide-gray-200 dark:divide-gray-700",
    dropdown: "border-gray-100 dark:border-gray-600 divide-gray-100 dark:divide-gray-600",
    navbar: "border-gray-100 dark:border-gray-700 divide-gray-100 dark:divide-gray-700",
    navbarUl: "border-gray-100 dark:border-gray-700 divide-gray-100 dark:divide-gray-700",
    form: "border-gray-300 dark:border-gray-700 divide-gray-300 dark:divide-gray-700",
    primary: "border-primary-500 dark:border-primary-200  divide-primary-500 dark:divide-primary-200 ",
    orange: "border-orange-300 dark:border-orange-800 divide-orange-300 dark:divide-orange-800",
    none: ""
  };
  let divClass;
  color = color ?? "default";
  setContext("color", color);
  divClass = twMerge(bgColors[color], textColors[color], rounded && "rounded-lg", border && "border", borderColors[color], shadow && "shadow-md", $$sanitized_props.class);
  if (transition && open) {
    $$payload.out += "<!--[-->";
    element(
      $$payload,
      tag,
      () => {
        $$payload.out += `${spread_attributes({ role, ...$$restProps, class: divClass })}`;
      },
      () => {
        $$payload.out += `<!---->`;
        slot($$payload, $$props, "default", {}, null);
        $$payload.out += `<!---->`;
      }
    );
  } else {
    $$payload.out += "<!--[!-->";
    if (open) {
      $$payload.out += "<!--[-->";
      element(
        $$payload,
        tag,
        () => {
          $$payload.out += `${spread_attributes({ role, ...$$restProps, class: divClass })}`;
        },
        () => {
          $$payload.out += `<!---->`;
          slot($$payload, $$props, "default", {}, null);
          $$payload.out += `<!---->`;
        }
      );
    } else {
      $$payload.out += "<!--[!-->";
    }
    $$payload.out += `<!--]-->`;
  }
  $$payload.out += `<!--]-->`;
  bind_props($$props, {
    tag,
    color,
    rounded,
    border,
    shadow,
    node,
    use,
    options,
    role,
    transition,
    params,
    open
  });
  pop();
}
function Button($$payload, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const $$restProps = rest_props($$sanitized_props, [
    "pill",
    "outline",
    "size",
    "href",
    "type",
    "color",
    "shadow",
    "tag",
    "checked",
    "disabled"
  ]);
  push();
  const group = getContext("group");
  let pill = fallback($$props["pill"], false);
  let outline = fallback($$props["outline"], false);
  let size = fallback($$props["size"], group ? "sm" : "md");
  let href = fallback($$props["href"], () => void 0, true);
  let type = fallback($$props["type"], "button");
  let color = fallback($$props["color"], group ? outline ? "dark" : "alternative" : "primary");
  let shadow = fallback($$props["shadow"], false);
  let tag = fallback($$props["tag"], "button");
  let checked = fallback($$props["checked"], () => void 0, true);
  let disabled = fallback($$props["disabled"], false);
  const colorClasses2 = {
    alternative: "text-gray-900 bg-white border border-gray-200 hover:bg-gray-100 dark:bg-gray-800 dark:text-gray-400 hover:text-primary-700 focus-within:text-primary-700 dark:focus-within:text-white dark:hover:text-white dark:hover:bg-gray-700",
    blue: "text-white bg-blue-700 hover:bg-blue-800 dark:bg-blue-600 dark:hover:bg-blue-700",
    dark: "text-white bg-gray-800 hover:bg-gray-900 dark:bg-gray-800 dark:hover:bg-gray-700",
    green: "text-white bg-green-700 hover:bg-green-800 dark:bg-green-600 dark:hover:bg-green-700",
    light: "text-gray-900 bg-white border border-gray-300 hover:bg-gray-100 dark:bg-gray-800 dark:text-white dark:border-gray-600 dark:hover:bg-gray-700 dark:hover:border-gray-600",
    primary: "text-white bg-primary-700 hover:bg-primary-800 dark:bg-primary-600 dark:hover:bg-primary-700",
    purple: "text-white bg-purple-700 hover:bg-purple-800 dark:bg-purple-600 dark:hover:bg-purple-700",
    red: "text-white bg-red-700 hover:bg-red-800 dark:bg-red-600 dark:hover:bg-red-700",
    yellow: "text-white bg-yellow-400 hover:bg-yellow-500 ",
    none: ""
  };
  const colorCheckedClasses = {
    alternative: "text-primary-700 border dark:text-primary-500 bg-gray-100 dark:bg-gray-700 border-gray-300 shadow-gray-300 dark:shadow-gray-800 shadow-inner",
    blue: "text-blue-900 bg-blue-400 dark:bg-blue-500 shadow-blue-700 dark:shadow-blue-800 shadow-inner",
    dark: "text-white bg-gray-500 dark:bg-gray-600 shadow-gray-800 dark:shadow-gray-900 shadow-inner",
    green: "text-green-900 bg-green-400 dark:bg-green-500 shadow-green-700 dark:shadow-green-800 shadow-inner",
    light: "text-gray-900 bg-gray-100 border border-gray-300 dark:bg-gray-500 dark:text-gray-900 dark:border-gray-700 shadow-gray-300 dark:shadow-gray-700 shadow-inner",
    primary: "text-primary-900 bg-primary-400 dark:bg-primary-500 shadow-primary-700 dark:shadow-primary-800 shadow-inner",
    purple: "text-purple-900 bg-purple-400 dark:bg-purple-500 shadow-purple-700 dark:shadow-purple-800 shadow-inner",
    red: "text-red-900 bg-red-400 dark:bg-red-500 shadow-red-700 dark:shadow-red-800 shadow-inner",
    yellow: "text-yellow-900 bg-yellow-300 dark:bg-yellow-400 shadow-yellow-500 dark:shadow-yellow-700 shadow-inner",
    none: ""
  };
  const coloredFocusClasses = {
    alternative: "focus-within:ring-gray-200 dark:focus-within:ring-gray-700",
    blue: "focus-within:ring-blue-300 dark:focus-within:ring-blue-800",
    dark: "focus-within:ring-gray-300 dark:focus-within:ring-gray-700",
    green: "focus-within:ring-green-300 dark:focus-within:ring-green-800",
    light: "focus-within:ring-gray-200 dark:focus-within:ring-gray-700",
    primary: "focus-within:ring-primary-300 dark:focus-within:ring-primary-800",
    purple: "focus-within:ring-purple-300 dark:focus-within:ring-purple-900",
    red: "focus-within:ring-red-300 dark:focus-within:ring-red-900",
    yellow: "focus-within:ring-yellow-300 dark:focus-within:ring-yellow-900",
    none: ""
  };
  const coloredShadowClasses = {
    alternative: "shadow-gray-500/50 dark:shadow-gray-800/80",
    blue: "shadow-blue-500/50 dark:shadow-blue-800/80",
    dark: "shadow-gray-500/50 dark:shadow-gray-800/80",
    green: "shadow-green-500/50 dark:shadow-green-800/80",
    light: "shadow-gray-500/50 dark:shadow-gray-800/80",
    primary: "shadow-primary-500/50 dark:shadow-primary-800/80",
    purple: "shadow-purple-500/50 dark:shadow-purple-800/80",
    red: "shadow-red-500/50 dark:shadow-red-800/80 ",
    yellow: "shadow-yellow-500/50 dark:shadow-yellow-800/80 ",
    none: ""
  };
  const outlineClasses = {
    alternative: "text-gray-900 dark:text-gray-400 hover:text-white border border-gray-800 hover:bg-gray-900 focus-within:bg-gray-900 focus-within:text-white focus-within:ring-gray-300 dark:border-gray-600 dark:hover:text-white dark:hover:bg-gray-600 dark:focus-within:ring-gray-800",
    blue: "text-blue-700 hover:text-white border border-blue-700 hover:bg-blue-800 dark:border-blue-500 dark:text-blue-500 dark:hover:text-white dark:hover:bg-blue-600",
    dark: "text-gray-900 hover:text-white border border-gray-800 hover:bg-gray-900 focus-within:bg-gray-900 focus-within:text-white dark:border-gray-600 dark:hover:text-white dark:hover:bg-gray-600",
    green: "text-green-700 hover:text-white border border-green-700 hover:bg-green-800 dark:border-green-500 dark:text-green-500 dark:hover:text-white dark:hover:bg-green-600",
    light: "text-gray-500 hover:text-gray-900 bg-white border border-gray-200 dark:border-gray-600 dark:hover:text-white dark:text-gray-400 hover:bg-gray-50 dark:bg-gray-700 dark:hover:bg-gray-600",
    primary: "text-primary-700 hover:text-white border border-primary-700 hover:bg-primary-700 dark:border-primary-500 dark:text-primary-500 dark:hover:text-white dark:hover:bg-primary-600",
    purple: "text-purple-700 hover:text-white border border-purple-700 hover:bg-purple-800 dark:border-purple-400 dark:text-purple-400 dark:hover:text-white dark:hover:bg-purple-500",
    red: "text-red-700 hover:text-white border border-red-700 hover:bg-red-800 dark:border-red-500 dark:text-red-500 dark:hover:text-white dark:hover:bg-red-600",
    yellow: "text-yellow-400 hover:text-white border border-yellow-400 hover:bg-yellow-500 dark:border-yellow-300 dark:text-yellow-300 dark:hover:text-white dark:hover:bg-yellow-400",
    none: ""
  };
  const sizeClasses = {
    xs: "px-3 py-2 text-xs",
    sm: "px-4 py-2 text-sm",
    md: "px-5 py-2.5 text-sm",
    lg: "px-5 py-3 text-base",
    xl: "px-6 py-3.5 text-base"
  };
  const hasBorder = () => outline || color === "alternative" || color === "light";
  let buttonClass;
  buttonClass = twMerge("text-center font-medium", group ? "focus-within:ring-2" : "focus-within:ring-4", group && "focus-within:z-10", group || "focus-within:outline-none", "inline-flex items-center justify-center " + sizeClasses[size], outline && checked && "border dark:border-gray-900", outline && checked && colorCheckedClasses[color], outline && !checked && outlineClasses[color], !outline && checked && colorCheckedClasses[color], !outline && !checked && colorClasses2[color], color === "alternative" && (group && !checked ? "dark:bg-gray-700 dark:text-white dark:border-gray-700 dark:hover:border-gray-600 dark:hover:bg-gray-600" : "dark:bg-transparent dark:border-gray-600 dark:hover:border-gray-600"), outline && color === "dark" && (group ? checked ? "bg-gray-900 border-gray-800 dark:border-white dark:bg-gray-600" : "dark:text-white border-gray-800 dark:border-white" : "dark:text-gray-400 dark:border-gray-700"), coloredFocusClasses[color], hasBorder() && group && "[&:not(:first-child)]:-ms-px", group ? pill && "first:rounded-s-full last:rounded-e-full" || "first:rounded-s-lg last:rounded-e-lg" : pill && "rounded-full" || "rounded-lg", shadow && "shadow-lg", shadow && coloredShadowClasses[color], disabled && "cursor-not-allowed opacity-50", $$sanitized_props.class);
  if (href && !disabled) {
    $$payload.out += "<!--[-->";
    $$payload.out += `<a${spread_attributes({
      href,
      ...$$restProps,
      class: buttonClass,
      role: "button"
    })}><!---->`;
    slot($$payload, $$props, "default", {}, null);
    $$payload.out += `<!----></a>`;
  } else {
    $$payload.out += "<!--[!-->";
    if (tag === "label") {
      $$payload.out += "<!--[-->";
      $$payload.out += `<label${spread_attributes({ ...$$restProps, class: buttonClass })}><!---->`;
      slot($$payload, $$props, "default", {}, null);
      $$payload.out += `<!----></label>`;
    } else {
      $$payload.out += "<!--[!-->";
      if (tag === "button") {
        $$payload.out += "<!--[-->";
        $$payload.out += `<button${spread_attributes({
          type,
          ...$$restProps,
          disabled,
          class: buttonClass
        })}><!---->`;
        slot($$payload, $$props, "default", {}, null);
        $$payload.out += `<!----></button>`;
      } else {
        $$payload.out += "<!--[!-->";
        element(
          $$payload,
          tag,
          () => {
            $$payload.out += `${spread_attributes({ ...$$restProps, class: buttonClass })}`;
          },
          () => {
            $$payload.out += `<!---->`;
            slot($$payload, $$props, "default", {}, null);
            $$payload.out += `<!---->`;
          }
        );
      }
      $$payload.out += `<!--]-->`;
    }
    $$payload.out += `<!--]-->`;
  }
  $$payload.out += `<!--]-->`;
  bind_props($$props, {
    pill,
    outline,
    size,
    href,
    type,
    color,
    shadow,
    tag,
    checked,
    disabled
  });
  pop();
}
function ButtonGroup($$payload, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const $$restProps = rest_props($$sanitized_props, ["size", "divClass"]);
  push();
  let size = fallback($$props["size"], "md");
  let divClass = fallback($$props["divClass"], "inline-flex rounded-lg shadow-sm");
  setContext("group", { size });
  $$payload.out += `<div${spread_attributes({
    ...$$restProps,
    class: twMerge(divClass, $$sanitized_props.class),
    role: "group"
  })}><!---->`;
  slot($$payload, $$props, "default", {}, null);
  $$payload.out += `<!----></div>`;
  bind_props($$props, { size, divClass });
  pop();
}
function Card($$payload, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const $$restProps = rest_props($$sanitized_props, [
    "href",
    "horizontal",
    "reverse",
    "img",
    "padding",
    "size",
    "imgClass"
  ]);
  push();
  let href = fallback($$props["href"], () => void 0, true);
  let horizontal = fallback($$props["horizontal"], false);
  let reverse = fallback($$props["reverse"], false);
  let img = fallback($$props["img"], () => void 0, true);
  let padding = fallback($$props["padding"], "lg");
  let size = fallback($$props["size"], "sm");
  let imgClass = fallback($$props["imgClass"], "");
  const paddings = {
    none: "",
    xs: "p-2",
    sm: "p-4",
    md: "p-4 sm:p-5",
    lg: "p-4 sm:p-6",
    xl: "p-4 sm:p-8"
  };
  const sizes = {
    none: "",
    xs: "max-w-xs",
    sm: "max-w-sm",
    md: "max-w-xl",
    lg: "max-w-2xl",
    xl: "max-w-screen-xl"
  };
  let innerPadding;
  let cardClass;
  let imgCls;
  innerPadding = paddings[padding];
  cardClass = twMerge("flex w-full", sizes[size], reverse ? "flex-col-reverse" : "flex-col", horizontal && (reverse ? "md:flex-row-reverse" : "md:flex-row"), href && "hover:bg-gray-100 dark:hover:bg-gray-700", !img && innerPadding, $$sanitized_props.class);
  imgCls = twMerge(reverse ? "rounded-b-lg" : "rounded-t-lg", horizontal && "object-cover w-full h-96 md:h-auto md:w-48 md:rounded-none", horizontal && (reverse ? "md:rounded-e-lg" : "md:rounded-s-lg"), imgClass);
  Frame($$payload, spread_props([
    {
      tag: href ? "a" : "div",
      rounded: true,
      shadow: true,
      border: true,
      href
    },
    $$restProps,
    {
      class: cardClass,
      children: ($$payload2) => {
        if (img) {
          $$payload2.out += "<!--[-->";
          $$payload2.out += `<img${attr("class", imgCls)}${attr("src", img)} alt=""> <div${attr("class", innerPadding)}><!---->`;
          slot($$payload2, $$props, "default", {}, null);
          $$payload2.out += `<!----></div>`;
        } else {
          $$payload2.out += "<!--[!-->";
          $$payload2.out += `<!---->`;
          slot($$payload2, $$props, "default", {}, null);
          $$payload2.out += `<!---->`;
        }
        $$payload2.out += `<!--]-->`;
      },
      $$slots: { default: true }
    }
  ]));
  bind_props($$props, {
    href,
    horizontal,
    reverse,
    img,
    padding,
    size,
    imgClass
  });
  pop();
}
function Popper($$payload, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const $$restProps = rest_props($$sanitized_props, [
    "activeContent",
    "arrow",
    "offset",
    "placement",
    "trigger",
    "triggeredBy",
    "reference",
    "strategy",
    "open",
    "yOnly",
    "middlewares"
  ]);
  push();
  let middleware;
  let activeContent = fallback($$props["activeContent"], false);
  let arrow = fallback($$props["arrow"], true);
  let offset = fallback($$props["offset"], 8);
  let placement = fallback($$props["placement"], "top");
  let trigger = fallback($$props["trigger"], "hover");
  let triggeredBy = fallback($$props["triggeredBy"], () => void 0, true);
  let reference = fallback($$props["reference"], () => void 0, true);
  let strategy = fallback($$props["strategy"], "absolute");
  let open = fallback($$props["open"], false);
  let yOnly = fallback($$props["yOnly"], false);
  let middlewares = fallback($$props["middlewares"], () => [dom.flip(), dom.shift()], true);
  let referenceEl;
  let floatingEl;
  let arrowEl;
  const px = (n) => n ? `${n}px` : "";
  let arrowSide;
  const oppositeSideMap = {
    left: "right",
    right: "left",
    bottom: "top",
    top: "bottom"
  };
  function updatePosition() {
    dom.computePosition(referenceEl, floatingEl, { placement, strategy, middleware }).then(({
      x,
      y,
      middlewareData,
      placement: placement2,
      strategy: strategy2
    }) => {
      floatingEl.style.position = strategy2;
      floatingEl.style.left = yOnly ? "0" : px(x);
      floatingEl.style.top = px(y);
      if (middlewareData.arrow && arrowEl instanceof HTMLDivElement) {
        arrowEl.style.left = px(middlewareData.arrow.x);
        arrowEl.style.top = px(middlewareData.arrow.y);
        arrowSide = oppositeSideMap[placement2.split("-")[0]];
        arrowEl.style[arrowSide] = px(-arrowEl.offsetWidth / 2 - ($$sanitized_props.border ? 1 : 0));
      }
    });
  }
  function init(node, _referenceEl) {
    floatingEl = node;
    let cleanup = dom.autoUpdate(_referenceEl, floatingEl, updatePosition);
    return {
      update(_referenceEl2) {
        cleanup();
        cleanup = dom.autoUpdate(_referenceEl2, floatingEl, updatePosition);
      },
      destroy() {
        cleanup();
      }
    };
  }
  let arrowClass;
  placement && (referenceEl = referenceEl);
  middleware = [
    ...middlewares,
    dom.offset(+offset),
    arrowEl
  ];
  arrowClass = twJoin("absolute pointer-events-none block w-[10px] h-[10px] rotate-45 bg-inherit border-inherit", $$sanitized_props.border && arrowSide === "bottom" && "border-b border-e", $$sanitized_props.border && arrowSide === "top" && "border-t border-s ", $$sanitized_props.border && arrowSide === "right" && "border-t border-e ", $$sanitized_props.border && arrowSide === "left" && "border-b border-s ");
  let $$settled = true;
  let $$inner_payload;
  function $$render_inner($$payload2) {
    if (!referenceEl) {
      $$payload2.out += "<!--[-->";
      $$payload2.out += `<div></div>`;
    } else {
      $$payload2.out += "<!--[!-->";
    }
    $$payload2.out += `<!--]--> `;
    if (referenceEl) {
      $$payload2.out += "<!--[-->";
      Frame($$payload2, spread_props([
        {
          use: init,
          options: referenceEl,
          get open() {
            return open;
          },
          set open($$value) {
            open = $$value;
            $$settled = false;
          },
          role: "tooltip",
          tabindex: activeContent ? -1 : void 0
        },
        $$restProps,
        {
          children: ($$payload3) => {
            $$payload3.out += `<!---->`;
            slot($$payload3, $$props, "default", {}, null);
            $$payload3.out += `<!----> `;
            if (arrow) {
              $$payload3.out += "<!--[-->";
              $$payload3.out += `<div${attr("class", arrowClass)}></div>`;
            } else {
              $$payload3.out += "<!--[!-->";
            }
            $$payload3.out += `<!--]-->`;
          },
          $$slots: { default: true }
        }
      ]));
    } else {
      $$payload2.out += "<!--[!-->";
    }
    $$payload2.out += `<!--]-->`;
  }
  do {
    $$settled = true;
    $$inner_payload = copy_payload($$payload);
    $$render_inner($$inner_payload);
  } while (!$$settled);
  assign_payload($$payload, $$inner_payload);
  bind_props($$props, {
    activeContent,
    arrow,
    offset,
    placement,
    trigger,
    triggeredBy,
    reference,
    strategy,
    open,
    yOnly,
    middlewares
  });
  pop();
}
function Dropdown($$payload, $$props) {
  const $$slots = sanitize_slots($$props);
  const $$sanitized_props = sanitize_props($$props);
  const $$restProps = rest_props($$sanitized_props, [
    "activeUrl",
    "open",
    "containerClass",
    "classContainer",
    "headerClass",
    "classHeader",
    "footerClass",
    "classFooter",
    "activeClass",
    "classActive",
    "arrow",
    "trigger",
    "placement",
    "color",
    "shadow",
    "rounded"
  ]);
  push();
  let containerCls, headerCls, ulCls, footerCls;
  let activeUrl = fallback($$props["activeUrl"], () => void 0, true);
  let open = fallback($$props["open"], false);
  let containerClass = fallback($$props["containerClass"], "divide-y z-50");
  let classContainer = fallback($$props["classContainer"], () => void 0, true);
  let headerClass = fallback($$props["headerClass"], "py-1 overflow-hidden rounded-t-lg");
  let classHeader = fallback($$props["classHeader"], () => void 0, true);
  let footerClass = fallback($$props["footerClass"], "py-1 overflow-hidden rounded-b-lg");
  let classFooter = fallback($$props["classFooter"], () => void 0, true);
  let activeClass = fallback($$props["activeClass"], "text-primary-700 dark:text-primary-700 hover:text-primary-900 dark:hover:text-primary-900");
  let classActive = fallback($$props["classActive"], () => void 0, true);
  let arrow = fallback($$props["arrow"], false);
  let trigger = fallback($$props["trigger"], "click");
  let placement = fallback($$props["placement"], "bottom");
  let color = fallback($$props["color"], "dropdown");
  let shadow = fallback($$props["shadow"], true);
  let rounded = fallback($$props["rounded"], true);
  const activeUrlStore = writable("");
  let activeCls = twMerge(activeClass, classActive);
  setContext("DropdownType", { activeClass: activeCls });
  setContext("activeUrl", activeUrlStore);
  activeUrlStore.set(activeUrl ?? "");
  containerCls = twMerge(containerClass, classContainer);
  headerCls = twMerge(headerClass, classHeader);
  ulCls = twMerge("py-1", $$sanitized_props.class);
  footerCls = twMerge(footerClass, classFooter);
  let $$settled = true;
  let $$inner_payload;
  function $$render_inner($$payload2) {
    Popper($$payload2, spread_props([
      { activeContent: true },
      $$restProps,
      {
        trigger,
        arrow,
        placement,
        shadow,
        rounded,
        color,
        class: containerCls,
        get open() {
          return open;
        },
        set open($$value) {
          open = $$value;
          $$settled = false;
        },
        children: ($$payload3) => {
          if ($$slots.header) {
            $$payload3.out += "<!--[-->";
            $$payload3.out += `<div${attr("class", headerCls)}><!---->`;
            slot($$payload3, $$props, "header", {}, null);
            $$payload3.out += `<!----></div>`;
          } else {
            $$payload3.out += "<!--[!-->";
          }
          $$payload3.out += `<!--]--> <ul${attr("class", ulCls)}><!---->`;
          slot($$payload3, $$props, "default", {}, null);
          $$payload3.out += `<!----></ul> `;
          if ($$slots.footer) {
            $$payload3.out += "<!--[-->";
            $$payload3.out += `<div${attr("class", footerCls)}><!---->`;
            slot($$payload3, $$props, "footer", {}, null);
            $$payload3.out += `<!----></div>`;
          } else {
            $$payload3.out += "<!--[!-->";
          }
          $$payload3.out += `<!--]-->`;
        },
        $$slots: { default: true }
      }
    ]));
  }
  do {
    $$settled = true;
    $$inner_payload = copy_payload($$payload);
    $$render_inner($$inner_payload);
  } while (!$$settled);
  assign_payload($$payload, $$inner_payload);
  bind_props($$props, {
    activeUrl,
    open,
    containerClass,
    classContainer,
    headerClass,
    classHeader,
    footerClass,
    classFooter,
    activeClass,
    classActive,
    arrow,
    trigger,
    placement,
    color,
    shadow,
    rounded
  });
  pop();
}
function DropdownDivider($$payload, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const $$restProps = rest_props($$sanitized_props, ["divClass"]);
  push();
  let divClass = fallback($$props["divClass"], "my-1 h-px bg-gray-100 dark:bg-gray-600");
  $$payload.out += `<div${spread_attributes({
    ...$$restProps,
    class: twMerge(divClass, $$sanitized_props.class)
  })}></div>`;
  bind_props($$props, { divClass });
  pop();
}
function Wrapper($$payload, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const $$restProps = rest_props($$sanitized_props, ["tag", "show", "use"]);
  let tag = fallback($$props["tag"], "div");
  let show = $$props["show"];
  let use = fallback($$props["use"], () => {
  });
  if (show) {
    $$payload.out += "<!--[-->";
    element(
      $$payload,
      tag,
      () => {
        $$payload.out += `${spread_attributes({ ...$$restProps })}`;
      },
      () => {
        $$payload.out += `<!---->`;
        slot($$payload, $$props, "default", {}, null);
        $$payload.out += `<!---->`;
      }
    );
  } else {
    $$payload.out += "<!--[!-->";
    $$payload.out += `<!---->`;
    slot($$payload, $$props, "default", {}, null);
    $$payload.out += `<!---->`;
  }
  $$payload.out += `<!--]-->`;
  bind_props($$props, { tag, show, use });
}
function DropdownItem($$payload, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const $$restProps = rest_props($$sanitized_props, ["defaultClass", "href", "activeClass"]);
  push();
  let active, liClass;
  let defaultClass = fallback($$props["defaultClass"], "font-medium py-2 px-4 text-sm hover:bg-gray-100 dark:hover:bg-gray-600");
  let href = fallback($$props["href"], () => void 0, true);
  let activeClass = fallback($$props["activeClass"], () => void 0, true);
  const context = getContext("DropdownType") ?? {};
  const activeUrlStore = getContext("activeUrl");
  let sidebarUrl = "";
  activeUrlStore.subscribe((value) => {
    sidebarUrl = value;
  });
  let wrap = true;
  function init(node) {
    wrap = node.parentElement?.tagName === "UL";
  }
  active = sidebarUrl ? href === sidebarUrl : false;
  liClass = twMerge(defaultClass, href ? "block" : "w-full text-left", active && (activeClass ?? context.activeClass), $$sanitized_props.class);
  Wrapper($$payload, {
    tag: "li",
    show: wrap,
    use: init,
    children: ($$payload2) => {
      const $$tag = href ? "a" : "button";
      element(
        $$payload2,
        $$tag,
        () => {
          $$payload2.out += `${spread_attributes({
            href,
            type: href ? void 0 : "button",
            role: href ? "link" : "button",
            ...$$restProps,
            class: liClass
          })}`;
        },
        () => {
          $$payload2.out += `<!---->`;
          slot($$payload2, $$props, "default", {}, null);
          $$payload2.out += `<!---->`;
        }
      );
    },
    $$slots: { default: true }
  });
  bind_props($$props, { defaultClass, href, activeClass });
  pop();
}
function Label($$payload, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const $$restProps = rest_props($$sanitized_props, ["color", "defaultClass", "show"]);
  push();
  let labelClass2;
  let color = fallback($$props["color"], "gray");
  let defaultClass = fallback($$props["defaultClass"], "text-sm rtl:text-right font-medium block");
  let show = fallback($$props["show"], true);
  const colorClasses2 = {
    gray: "text-gray-900 dark:text-gray-300",
    green: "text-green-700 dark:text-green-500",
    red: "text-red-700 dark:text-red-500",
    disabled: "text-gray-400 dark:text-gray-500 grayscale contrast-50"
  };
  {
    color = color;
  }
  labelClass2 = twMerge(defaultClass, colorClasses2[color], $$sanitized_props.class);
  if (show) {
    $$payload.out += "<!--[-->";
    $$payload.out += `<label${spread_attributes({ ...$$restProps, class: labelClass2 })}><!---->`;
    slot($$payload, $$props, "default", {}, null);
    $$payload.out += `<!----></label>`;
  } else {
    $$payload.out += "<!--[!-->";
    $$payload.out += `<!---->`;
    slot($$payload, $$props, "default", {}, null);
    $$payload.out += `<!---->`;
  }
  $$payload.out += `<!--]-->`;
  bind_props($$props, { color, defaultClass, show });
  pop();
}
const colorClasses = {
  primary: "text-primary-600 focus:ring-primary-500 dark:focus:ring-primary-600",
  secondary: "text-secondary-600 focus:ring-secondary-500 dark:focus:ring-secondary-600",
  red: "text-red-600 focus:ring-red-500 dark:focus:ring-red-600",
  green: "text-green-600 focus:ring-green-500 dark:focus:ring-green-600",
  purple: "text-purple-600 focus:ring-purple-500 dark:focus:ring-purple-600",
  teal: "text-teal-600 focus:ring-teal-500 dark:focus:ring-teal-600",
  yellow: "text-yellow-400 focus:ring-yellow-500 dark:focus:ring-yellow-600",
  orange: "text-orange-500 focus:ring-orange-500 dark:focus:ring-orange-600",
  blue: "text-blue-600 focus:ring-blue-500 dark:focus:ring-blue-600"
};
const labelClass = (inline, extraClass) => twMerge(inline ? "inline-flex" : "flex", "items-center", extraClass);
const inputClass = (custom, color, rounded, tinted, spacing, extraClass) => twMerge("w-4 h-4 bg-gray-100 border-gray-300 dark:ring-offset-gray-800 focus:ring-2", spacing, tinted ? "dark:bg-gray-600 dark:border-gray-500" : "dark:bg-gray-700 dark:border-gray-600", custom && "sr-only peer", rounded && "rounded", colorClasses[color], extraClass);
function Radio($$payload, $$props) {
  const $$slots = sanitize_slots($$props);
  const $$sanitized_props = sanitize_props($$props);
  const $$restProps = rest_props($$sanitized_props, [
    "color",
    "custom",
    "inline",
    "group",
    "value",
    "spacing",
    "checked"
  ]);
  push();
  let color = fallback($$props["color"], "primary");
  let custom = fallback($$props["custom"], false);
  let inline = fallback($$props["inline"], false);
  let group = fallback($$props["group"], () => void 0, true);
  let value = fallback($$props["value"], "");
  let spacing = fallback($$props["spacing"], () => $$slots.default ? "me-2" : "", true);
  let checked = fallback($$props["checked"], false);
  let background = getContext("background");
  if (checked && group === void 0) {
    group = value;
  }
  Label($$payload, {
    class: labelClass(inline, $$sanitized_props.class),
    show: $$slots.default,
    children: ($$payload2) => {
      $$payload2.out += `<input${spread_attributes({
        type: "radio",
        ...$$restProps,
        checked: group === value,
        value,
        class: inputClass(custom, color, false, background, spacing, $$slots.default || $$sanitized_props.class)
      })}> <!---->`;
      slot($$payload2, $$props, "default", {}, null);
      $$payload2.out += `<!---->`;
    },
    $$slots: { default: true }
  });
  bind_props($$props, {
    color,
    custom,
    inline,
    group,
    value,
    spacing,
    checked
  });
  pop();
}
function Checkbox($$payload, $$props) {
  const $$slots = sanitize_slots($$props);
  const $$sanitized_props = sanitize_props($$props);
  const $$restProps = rest_props($$sanitized_props, [
    "color",
    "custom",
    "inline",
    "group",
    "choices",
    "value",
    "checked",
    "spacing",
    "groupLabelClass",
    "groupInputClass"
  ]);
  push();
  let color = fallback($$props["color"], "primary");
  let custom = fallback($$props["custom"], false);
  let inline = fallback($$props["inline"], false);
  let group = fallback($$props["group"], () => [], true);
  let choices = fallback($$props["choices"], () => [], true);
  let value = fallback($$props["value"], "on");
  let checked = fallback($$props["checked"], () => void 0, true);
  let spacing = fallback($$props["spacing"], () => $$slots.default ? "me-2" : "", true);
  let groupLabelClass = fallback($$props["groupLabelClass"], "");
  let groupInputClass = fallback($$props["groupInputClass"], "");
  let background = getContext("background");
  if (choices.length > 0) {
    $$payload.out += "<!--[-->";
    const each_array = ensure_array_like(choices);
    $$payload.out += `<!--[-->`;
    for (let i = 0, $$length = each_array.length; i < $$length; i++) {
      let { value: value2, label } = each_array[i];
      Label($$payload, {
        class: labelClass(inline, groupLabelClass),
        show: $$slots.default,
        for: `checkbox-${i}`,
        children: ($$payload2) => {
          $$payload2.out += `<!---->${escape_html(label)} <input${spread_attributes({
            id: `checkbox-${i}`,
            type: "checkbox",
            value: value2,
            checked: group.includes(value2),
            ...$$restProps,
            class: inputClass(custom, color, true, background, spacing, groupInputClass)
          })}> <!---->`;
          slot($$payload2, $$props, "default", {}, null);
          $$payload2.out += `<!---->`;
        },
        $$slots: { default: true }
      });
    }
    $$payload.out += `<!--]-->`;
  } else {
    $$payload.out += "<!--[!-->";
    Label($$payload, {
      class: labelClass(inline, $$sanitized_props.class),
      show: $$slots.default,
      children: ($$payload2) => {
        $$payload2.out += `<input${spread_attributes({
          type: "checkbox",
          checked,
          value,
          ...$$restProps,
          class: inputClass(custom, color, true, background, spacing, $$slots.default || $$sanitized_props.class)
        })}> <!---->`;
        slot($$payload2, $$props, "default", {}, null);
        $$payload2.out += `<!---->`;
      },
      $$slots: { default: true }
    });
  }
  $$payload.out += `<!--]-->`;
  bind_props($$props, {
    color,
    custom,
    inline,
    group,
    choices,
    value,
    checked,
    spacing,
    groupLabelClass,
    groupInputClass
  });
  pop();
}
function clampSize(s) {
  return s && s === "xs" ? "sm" : s === "xl" ? "lg" : s;
}
function Input($$payload, $$props) {
  const $$slots = sanitize_slots($$props);
  const $$sanitized_props = sanitize_props($$props);
  const $$restProps = rest_props($$sanitized_props, [
    "type",
    "value",
    "size",
    "clearable",
    "defaultClass",
    "color",
    "floatClass",
    "classLeft",
    "classRight"
  ]);
  push();
  let _size;
  let type = fallback($$props["type"], "text");
  let value = fallback($$props["value"], () => void 0, true);
  let size = fallback($$props["size"], () => void 0, true);
  let clearable = fallback($$props["clearable"], false);
  let defaultClass = fallback($$props["defaultClass"], "block w-full disabled:cursor-not-allowed disabled:opacity-50 rtl:text-right");
  let color = fallback($$props["color"], "base");
  let floatClass = fallback($$props["floatClass"], "flex absolute inset-y-0 items-center text-gray-500 dark:text-gray-400");
  let classLeft = fallback($$props["classLeft"], "");
  let classRight = fallback($$props["classRight"], "");
  const borderClasses = {
    base: "border border-gray-300 dark:border-gray-600",
    tinted: "border border-gray-300 dark:border-gray-500",
    green: "border border-green-500 dark:border-green-400",
    red: "border border-red-500 dark:border-red-400"
  };
  const ringClasses = {
    base: "focus:border-primary-500 focus:ring-primary-500 dark:focus:border-primary-500 dark:focus:ring-primary-500",
    green: "focus:ring-green-500 focus:border-green-500 dark:focus:border-green-500 dark:focus:ring-green-500",
    red: "focus:ring-red-500 focus:border-red-500 dark:focus:ring-red-500 dark:focus:border-red-500"
  };
  const colorClasses2 = {
    base: "bg-gray-50 text-gray-900 dark:bg-gray-700 dark:text-white dark:placeholder-gray-400",
    tinted: "bg-gray-50 text-gray-900 dark:bg-gray-600 dark:text-white dark:placeholder-gray-400",
    green: "bg-green-50 text-green-900 placeholder-green-700 dark:text-green-400 dark:placeholder-green-500 dark:bg-gray-700",
    red: "bg-red-50 text-red-900 placeholder-red-700 dark:text-red-500 dark:placeholder-red-500 dark:bg-gray-700"
  };
  let background = getContext("background");
  let group = getContext("group");
  const textSizes = {
    sm: "sm:text-xs",
    md: "text-sm",
    lg: "sm:text-base"
  };
  const leftPadding = { sm: "ps-9", md: "ps-10", lg: "ps-11" };
  const rightPadding = { sm: "pe-9", md: "pe-10", lg: "pe-11" };
  const inputPadding = { sm: "p-2", md: "p-2.5", lg: "p-3" };
  let inputClass2;
  _size = size || clampSize(group?.size) || "md";
  {
    const _color = color === "base" && background ? "tinted" : color;
    inputClass2 = twMerge([
      defaultClass,
      inputPadding[_size],
      $$slots.left && leftPadding[_size] || (clearable || $$slots.right) && rightPadding[_size],
      ringClasses[color],
      colorClasses2[_color],
      borderClasses[_color],
      textSizes[_size],
      group || "rounded-lg",
      group && "first:rounded-s-lg last:rounded-e-lg",
      group && "[&:not(:first-child)]:-ms-px",
      $$sanitized_props.class
    ]);
  }
  Wrapper($$payload, {
    class: "relative w-full",
    show: $$slots.left || $$slots.right,
    children: ($$payload2) => {
      if ($$slots.left) {
        $$payload2.out += "<!--[-->";
        $$payload2.out += `<div${attr("class", `${stringify(twMerge(floatClass, classLeft))} start-0 ps-2.5 pointer-events-none`)}><!---->`;
        slot($$payload2, $$props, "left", {}, null);
        $$payload2.out += `<!----></div>`;
      } else {
        $$payload2.out += "<!--[!-->";
      }
      $$payload2.out += `<!--]--> <!---->`;
      slot(
        $$payload2,
        $$props,
        "default",
        {
          props: { ...$$restProps, class: inputClass2 }
        },
        () => {
          $$payload2.out += `<input${spread_attributes({
            ...$$restProps,
            value,
            ...{ type },
            class: inputClass2
          })}>`;
        }
      );
      $$payload2.out += `<!----> `;
      if ($$slots.right) {
        $$payload2.out += "<!--[-->";
        $$payload2.out += `<div${attr("class", `${stringify(twMerge(floatClass, classRight))} end-0 pe-2.5`)}><!---->`;
        slot($$payload2, $$props, "right", {}, null);
        $$payload2.out += `<!----></div>`;
      } else {
        $$payload2.out += "<!--[!-->";
      }
      $$payload2.out += `<!--]--> `;
      if (clearable && value && `${value}`.length > 0) {
        $$payload2.out += "<!--[-->";
        CloseButton($$payload2, {
          size,
          color: "none",
          class: ` ${stringify(twMerge(floatClass, classRight))} focus:ring-0 end-6 focus:ring-gray-400 dark:text-white`
        });
      } else {
        $$payload2.out += "<!--[!-->";
      }
      $$payload2.out += `<!--]-->`;
    },
    $$slots: { default: true }
  });
  bind_props($$props, {
    type,
    value,
    size,
    clearable,
    defaultClass,
    color,
    floatClass,
    classLeft,
    classRight
  });
  pop();
}
function InputAddon($$payload, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const $$restProps = rest_props($$sanitized_props, ["size"]);
  push();
  let _size, divClass;
  let size = fallback($$props["size"], () => void 0, true);
  let background = getContext("background");
  let group = getContext("group");
  const borderClasses = {
    base: "border-gray-300 dark:border-gray-600",
    tinted: "border-gray-300 dark:border-gray-500"
  };
  const darkBgClasses = {
    base: "dark:bg-gray-600 dark:text-gray-400",
    tinted: "dark:bg-gray-500 dark:text-gray-300"
  };
  const divider = {
    base: "dark:border-e-gray-700 dark:last:border-e-gray-600",
    tinted: "dark:border-e-gray-600 dark:last:border-e-gray-500"
  };
  const textSizes = {
    sm: "sm:text-xs",
    md: "text-sm",
    lg: "sm:text-base"
  };
  const prefixPadding = { sm: "px-2", md: "px-3", lg: "px-4" };
  _size = size || clampSize(group?.size) || "md";
  divClass = twMerge(textSizes[_size], prefixPadding[_size], "text-gray-500 bg-gray-200", background ? darkBgClasses.tinted : darkBgClasses.base, background ? divider.tinted : divider.base, background ? borderClasses["tinted"] : borderClasses["base"], "inline-flex items-center border", group && "[&:not(:first-child)]:-ms-px", "first:rounded-s-lg last:rounded-e-lg", $$sanitized_props.class);
  $$payload.out += `<div${spread_attributes({ ...$$restProps, class: divClass })}><!---->`;
  slot($$payload, $$props, "default", {}, null);
  $$payload.out += `<!----></div>`;
  bind_props($$props, { size });
  pop();
}
function Toggle($$payload, $$props) {
  const $$slots = sanitize_slots($$props);
  const $$sanitized_props = sanitize_props($$props);
  const $$restProps = rest_props($$sanitized_props, [
    "size",
    "group",
    "value",
    "checked",
    "customSize",
    "classDiv",
    "disabled"
  ]);
  push();
  let size = fallback($$props["size"], "default");
  let group = fallback($$props["group"], () => [], true);
  let value = fallback($$props["value"], "");
  let checked = fallback($$props["checked"], () => void 0, true);
  let customSize = fallback($$props["customSize"], "");
  let classDiv = fallback($$props["classDiv"], "");
  let disabled = fallback($$props["disabled"], false);
  let background = getContext("background");
  const common = "me-3 shrink-0 bg-gray-200 rounded-full peer-focus:ring-4 peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:bg-white after:border-gray-300 after:border after:rounded-full after:transition-all";
  const colors = {
    primary: "peer-focus:ring-primary-300 dark:peer-focus:ring-primary-800 peer-checked:bg-primary-600",
    secondary: "peer-focus:ring-secondary-300 dark:peer-focus:ring-secondary-800 peer-checked:bg-secondary-600",
    red: "peer-focus:ring-red-300 dark:peer-focus:ring-red-800 peer-checked:bg-red-600",
    green: "peer-focus:ring-green-300 dark:peer-focus:ring-green-800 peer-checked:bg-green-600",
    purple: "peer-focus:ring-purple-300 dark:peer-focus:ring-purple-800 peer-checked:bg-purple-600",
    yellow: "peer-focus:ring-yellow-300 dark:peer-focus:ring-yellow-800 peer-checked:bg-yellow-400",
    teal: "peer-focus:ring-teal-300 dark:peer-focus:ring-teal-800 peer-checked:bg-teal-600",
    orange: "peer-focus:ring-orange-300 dark:peer-focus:ring-orange-800 peer-checked:bg-orange-500",
    blue: "peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 peer-checked:bg-blue-600"
  };
  const sizes = {
    small: "w-9 h-5 after:top-[2px] after:start-[2px] after:h-4 after:w-4",
    default: "w-11 h-6 after:top-0.5 after:start-[2px] after:h-5 after:w-5",
    large: "w-14 h-7 after:top-0.5 after:start-[4px]  after:h-6 after:w-6",
    custom: customSize
  };
  let divClass;
  let checkboxCls;
  divClass = twMerge(common, $$slots.offLabel ? "ms-3" : "", background ? "dark:bg-gray-600 dark:border-gray-500" : "dark:bg-gray-700 dark:border-gray-600", colors[$$restProps.color ?? "primary"], sizes[size], "relative", classDiv);
  checkboxCls = disabled ? "cursor-not-allowed grayscale contrast-50 text-gray-400" : "cursor-pointer text-gray-900";
  let $$settled = true;
  let $$inner_payload;
  function $$render_inner($$payload2) {
    Checkbox($$payload2, spread_props([
      { custom: true },
      $$restProps,
      {
        disabled,
        class: twMerge(checkboxCls, $$sanitized_props.class),
        value,
        get checked() {
          return checked;
        },
        set checked($$value) {
          checked = $$value;
          $$settled = false;
        },
        get group() {
          return group;
        },
        set group($$value) {
          group = $$value;
          $$settled = false;
        },
        children: ($$payload3) => {
          $$payload3.out += `<!---->`;
          slot($$payload3, $$props, "offLabel", {}, null);
          $$payload3.out += `<!----> <span${attr("class", divClass)}></span> <!---->`;
          slot($$payload3, $$props, "default", {}, null);
          $$payload3.out += `<!---->`;
        },
        $$slots: { default: true }
      }
    ]));
  }
  do {
    $$settled = true;
    $$inner_payload = copy_payload($$payload);
    $$render_inner($$inner_payload);
  } while (!$$settled);
  assign_payload($$payload, $$inner_payload);
  bind_props($$props, {
    size,
    group,
    value,
    checked,
    customSize,
    classDiv,
    disabled
  });
  pop();
}
function linear(t) {
  return t;
}
function cubicOut(t) {
  const f = t - 1;
  return f * f * f + 1;
}
function is_date(obj) {
  return Object.prototype.toString.call(obj) === "[object Date]";
}
function get_interpolator(a, b) {
  if (a === b || a !== a) return () => a;
  const type = typeof a;
  if (type !== typeof b || Array.isArray(a) !== Array.isArray(b)) {
    throw new Error("Cannot interpolate values of different type");
  }
  if (Array.isArray(a)) {
    const arr = (
      /** @type {Array<any>} */
      b.map((bi, i) => {
        return get_interpolator(
          /** @type {Array<any>} */
          a[i],
          bi
        );
      })
    );
    return (t) => arr.map((fn) => fn(t));
  }
  if (type === "object") {
    if (!a || !b) {
      throw new Error("Object cannot be null");
    }
    if (is_date(a) && is_date(b)) {
      const an = a.getTime();
      const bn = b.getTime();
      const delta = bn - an;
      return (t) => new Date(an + t * delta);
    }
    const keys = Object.keys(b);
    const interpolators = {};
    keys.forEach((key) => {
      interpolators[key] = get_interpolator(a[key], b[key]);
    });
    return (t) => {
      const result = {};
      keys.forEach((key) => {
        result[key] = interpolators[key](t);
      });
      return result;
    };
  }
  if (type === "number") {
    const delta = (
      /** @type {number} */
      b - /** @type {number} */
      a
    );
    return (t) => a + t * delta;
  }
  throw new Error(`Cannot interpolate ${type} values`);
}
function tweened(value, defaults = {}) {
  const store = writable(value);
  let task;
  let target_value = value;
  function set(new_value, opts) {
    target_value = new_value;
    if (value == null) {
      store.set(value = new_value);
      return Promise.resolve();
    }
    let previous_task = task;
    let started = false;
    let {
      delay = 0,
      duration = 400,
      easing = linear,
      interpolate = get_interpolator
    } = { ...defaults, ...opts };
    if (duration === 0) {
      if (previous_task) {
        previous_task.abort();
        previous_task = null;
      }
      store.set(value = target_value);
      return Promise.resolve();
    }
    const start = raf.now() + delay;
    let fn;
    task = loop((now2) => {
      if (now2 < start) return true;
      if (!started) {
        fn = interpolate(
          /** @type {any} */
          value,
          new_value
        );
        if (typeof duration === "function")
          duration = duration(
            /** @type {any} */
            value,
            new_value
          );
        started = true;
      }
      if (previous_task) {
        previous_task.abort();
        previous_task = null;
      }
      const elapsed = now2 - start;
      if (elapsed > /** @type {number} */
      duration) {
        store.set(value = new_value);
        return false;
      }
      store.set(value = fn(easing(elapsed / duration)));
      return true;
    });
    return task.promise;
  }
  return {
    set,
    update: (fn, opts) => set(fn(
      /** @type {any} */
      target_value,
      /** @type {any} */
      value
    ), opts),
    subscribe: store.subscribe
  };
}
function Progressbar($$payload, $$props) {
  const $$sanitized_props = sanitize_props($$props);
  const $$restProps = rest_props($$sanitized_props, [
    "progress",
    "precision",
    "tweenDuration",
    "animate",
    "size",
    "labelInside",
    "labelOutside",
    "easing",
    "color",
    "labelInsideClass",
    "divClass",
    "progressClass",
    "classLabelOutside"
  ]);
  push();
  var $$store_subs;
  let progress = fallback($$props["progress"], "45");
  let precision = fallback($$props["precision"], 0);
  let tweenDuration = fallback($$props["tweenDuration"], 400);
  let animate = fallback($$props["animate"], false);
  let size = fallback($$props["size"], "h-2.5");
  let labelInside = fallback($$props["labelInside"], false);
  let labelOutside = fallback($$props["labelOutside"], "");
  let easing = fallback($$props["easing"], cubicOut);
  let color = fallback($$props["color"], "primary");
  let labelInsideClass = fallback($$props["labelInsideClass"], "text-primary-100 text-xs font-medium text-center p-0.5 leading-none rounded-full");
  let divClass = fallback($$props["divClass"], "w-full bg-gray-200 rounded-full dark:bg-gray-700");
  let progressClass = fallback($$props["progressClass"], "");
  let classLabelOutside = fallback($$props["classLabelOutside"], "");
  const _progress = tweened(0, { duration: animate ? tweenDuration : 0, easing });
  const barColors = {
    primary: "bg-primary-600",
    blue: "bg-blue-600",
    gray: "bg-gray-600 dark:bg-gray-300",
    red: "bg-red-600 dark:bg-red-500",
    green: "bg-green-600 dark:bg-green-500",
    yellow: "bg-yellow-400",
    purple: "bg-purple-600 dark:bg-purple-500",
    indigo: "bg-indigo-600 dark:bg-indigo-500"
  };
  _progress.set(Number(progress));
  if (labelOutside) {
    $$payload.out += "<!--[-->";
    $$payload.out += `<div${spread_attributes({
      ...$$restProps,
      class: twMerge("flex justify-between mb-1", classLabelOutside)
    })}><span class="text-base font-medium text-blue-700 dark:text-white">${escape_html(labelOutside)}</span> <span class="text-sm font-medium text-blue-700 dark:text-white">${escape_html(progress)}%</span></div>`;
  } else {
    $$payload.out += "<!--[!-->";
  }
  $$payload.out += `<!--]--> <div${attr("class", twMerge(divClass, size, $$sanitized_props.class))}>`;
  if (labelInside) {
    $$payload.out += "<!--[-->";
    $$payload.out += `<div${attr("class", twMerge(barColors[color], labelInsideClass))}${attr("style", `width: ${stringify(store_get($$store_subs ??= {}, "$_progress", _progress))}%`)}>${escape_html(store_get($$store_subs ??= {}, "$_progress", _progress).toFixed(precision))}%</div>`;
  } else {
    $$payload.out += "<!--[!-->";
    $$payload.out += `<div${attr("class", twMerge(barColors[color], size, "rounded-full", progressClass))}${attr("style", `width: ${stringify(store_get($$store_subs ??= {}, "$_progress", _progress))}%`)}></div>`;
  }
  $$payload.out += `<!--]--></div>`;
  if ($$store_subs) unsubscribe_stores($$store_subs);
  bind_props($$props, {
    progress,
    precision,
    tweenDuration,
    animate,
    size,
    labelInside,
    labelOutside,
    easing,
    color,
    labelInsideClass,
    divClass,
    progressClass,
    classLabelOutside
  });
  pop();
}
function ChevronDownOutline($$payload, $$props) {
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
    strokeWidth = ctx.strokeWidth || "2",
    desc,
    class: className,
    ariaLabel = "chevron down outline",
    $$slots,
    $$events,
    ...restProps
  } = $$props;
  let ariaDescribedby = `${title?.id || ""} ${desc?.id || ""}`;
  const hasDescription = !!(title?.id || desc?.id);
  $$payload.out += `<svg${spread_attributes(
    {
      xmlns: "http://www.w3.org/2000/svg",
      fill: "none",
      color,
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
  $$payload.out += `<!--]--><path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"${attr("stroke-width", strokeWidth)} d="m8 10 4 4 4-4"></path></svg>`;
  pop();
}
function ChevronRightOutline($$payload, $$props) {
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
    strokeWidth = ctx.strokeWidth || "2",
    desc,
    class: className,
    ariaLabel = "chevron right outline",
    $$slots,
    $$events,
    ...restProps
  } = $$props;
  let ariaDescribedby = `${title?.id || ""} ${desc?.id || ""}`;
  const hasDescription = !!(title?.id || desc?.id);
  $$payload.out += `<svg${spread_attributes(
    {
      xmlns: "http://www.w3.org/2000/svg",
      fill: "none",
      color,
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
  $$payload.out += `<!--]--><path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"${attr("stroke-width", strokeWidth)} d="m10 16 4-4-4-4"></path></svg>`;
  pop();
}
function CloseCircleOutline($$payload, $$props) {
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
    strokeWidth = ctx.strokeWidth || "2",
    desc,
    class: className,
    ariaLabel = "close circle outline",
    $$slots,
    $$events,
    ...restProps
  } = $$props;
  let ariaDescribedby = `${title?.id || ""} ${desc?.id || ""}`;
  const hasDescription = !!(title?.id || desc?.id);
  $$payload.out += `<svg${spread_attributes(
    {
      xmlns: "http://www.w3.org/2000/svg",
      fill: "none",
      color,
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
  $$payload.out += `<!--]--><path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"${attr("stroke-width", strokeWidth)} d="m15 9-6 6m0-6 6 6m6-3a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"></path></svg>`;
  pop();
}
function PauseOutline($$payload, $$props) {
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
    strokeWidth = ctx.strokeWidth || "2",
    desc,
    class: className,
    ariaLabel = "pause outline",
    $$slots,
    $$events,
    ...restProps
  } = $$props;
  let ariaDescribedby = `${title?.id || ""} ${desc?.id || ""}`;
  const hasDescription = !!(title?.id || desc?.id);
  $$payload.out += `<svg${spread_attributes(
    {
      xmlns: "http://www.w3.org/2000/svg",
      fill: "none",
      color,
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
  $$payload.out += `<!--]--><path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"${attr("stroke-width", strokeWidth)} d="M9 6H8a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h1a1 1 0 0 0 1-1V7a1 1 0 0 0-1-1Zm7 0h-1a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h1a1 1 0 0 0 1-1V7a1 1 0 0 0-1-1Z"></path></svg>`;
  pop();
}
function PlayOutline($$payload, $$props) {
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
    strokeWidth = ctx.strokeWidth || "2",
    desc,
    class: className,
    ariaLabel = "play outline",
    $$slots,
    $$events,
    ...restProps
  } = $$props;
  let ariaDescribedby = `${title?.id || ""} ${desc?.id || ""}`;
  const hasDescription = !!(title?.id || desc?.id);
  $$payload.out += `<svg${spread_attributes(
    {
      xmlns: "http://www.w3.org/2000/svg",
      fill: "none",
      color,
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
  $$payload.out += `<!--]--><path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"${attr("stroke-width", strokeWidth)} d="M8 18V6l8 6-8 6Z"></path></svg>`;
  pop();
}
const previewState = writable({
  quantity: "",
  unit: "",
  component: "",
  layer: 0,
  maxPoints: 0,
  type: "",
  vectorFieldValues: [],
  vectorFieldPositions: [],
  scalarField: [],
  min: 0,
  max: 0,
  refresh: false,
  nComp: 0,
  dataPointsCount: 0,
  xPossibleSizes: [],
  yPossibleSizes: [],
  xChosenSize: 0,
  yChosenSize: 0,
  dynQuantities: {},
  startX: 0,
  startY: 0,
  startZ: 0,
  symmetricX: false,
  symmetricY: false,
  dynQuantitiesCat: []
});
const headerState = writable({
  path: "",
  status: "",
  version: ""
});
const solverState = writable({
  type: "",
  steps: 0,
  time: 0,
  dt: 0,
  errPerStep: 0,
  maxTorque: 0,
  fixdt: 0,
  mindt: 0,
  maxdt: 0,
  maxerr: 0,
  nundone: 0
});
const consoleState = writable({ hist: "" });
const meshState = writable({
  dx: 0,
  dy: 0,
  dz: 0,
  Nx: 0,
  Ny: 0,
  Nz: 0,
  Tx: 0,
  Ty: 0,
  Tz: 0,
  PBCx: 0,
  PBCy: 0,
  PBCz: 0
});
const parametersState = writable({
  regions: [],
  fields: [],
  selectedRegion: 0
});
const tablePlotState = writable({
  autoSaveInterval: 0,
  columns: [],
  xColumn: "t",
  yColumn: "mx",
  xColumnUnit: "s",
  yColumnUnit: "",
  data: [],
  xmin: 0,
  xmax: 0,
  ymin: 0,
  ymax: 0,
  maxPoints: 0,
  step: 0
});
async function post(endpoint, data) {
  const response = await fetch(`./api/${endpoint}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ ...data })
  });
  if (!response.ok) {
    try {
      const errorData = await response.json();
      setAlert(errorData.error);
    } catch (e) {
      setAlert("An error occurred");
    }
  }
}
function postLayer(layer) {
  post("preview/layer", { layer });
}
function postXChosenSize(xChosenSize) {
  post("preview/XChosenSize", { xChosenSize });
}
function postYChosenSize(yChosenSize) {
  post("preview/YChosenSize", { yChosenSize });
}
const metricsState = writable({
  pid: 0,
  error: "",
  cpuPercent: 0,
  cpuPercentTotal: 0,
  ramPercent: 0,
  ramPercentTotal: 0,
  gpuName: "",
  gpuUtilizationPercent: 0,
  gpuUUID: "",
  gpuTemperature: 0,
  gpuPowerDraw: 0,
  gpuPowerLimit: 0,
  gpuVramUsed: 0,
  gpuVramTotal: 0
});
let connected = writable(false);
function Header($$payload, $$props) {
  push();
  var $$store_subs;
  $$payload.out += `<section class="sticky top-0 z-10 m-2.5 mb-0 p-2"><div class="flex w-screen items-center gap-2"><div class="text-3xl">`;
  if (store_get($$store_subs ??= {}, "$connected", connected)) {
    $$payload.out += "<!--[-->";
    if (store_get($$store_subs ??= {}, "$headerState", headerState).status === "running") {
      $$payload.out += "<!--[-->";
      PlayOutline($$payload, { color: "green", size: "xl" });
    } else {
      $$payload.out += "<!--[!-->";
      if (store_get($$store_subs ??= {}, "$headerState", headerState).status === "paused") {
        $$payload.out += "<!--[-->";
        PauseOutline($$payload, { color: "orange", size: "xl" });
      } else {
        $$payload.out += "<!--[!-->";
        CloseCircleOutline($$payload, { color: "red", size: "xl" });
      }
      $$payload.out += `<!--]-->`;
    }
    $$payload.out += `<!--]-->`;
  } else {
    $$payload.out += "<!--[!-->";
    CloseCircleOutline($$payload, { color: "red", size: "xl" });
  }
  $$payload.out += `<!--]--></div> <div class="min-w-0 flex-grow truncate text-3xl text-green-500">${escape_html(store_get($$store_subs ??= {}, "$headerState", headerState).path)}</div> <div class="w-40 whitespace-nowrap text-xl text-gray-500">v.∞</div></div></section>`;
  if ($$store_subs) unsubscribe_stores($$store_subs);
  pop();
}
const quantities = {
  Common: ["m", "torque", "regions", "Msat", "Aex", "alpha"],
  "Magnetic Fields": [
    "B_anis",
    "B_custom",
    "B_demag",
    "B_eff",
    "B_exch",
    "B_ext",
    "B_mel",
    "B_therm"
  ],
  Energy: [
    "Edens_anis",
    "Edens_custom",
    "Edens_demag",
    "Edens_exch",
    "Edens_mel",
    "Edens_therm",
    "Edens_total",
    "Edens_Zeeman",
    "Edens_el",
    "Edens_kin"
  ],
  "Force": [
    "F_melM",
    "F_el",
    "F_elsys",
    "rhod2udt2",
    "etadudt"
  ],
  Anisotropy: ["anisC1", "anisC2", "anisU", "Kc1", "Kc2", "Kc3", "Ku1", "Ku2"],
  DMI: ["Dbulk", "Dind", "DindCoupling"],
  External: [
    "ext_bubbledist",
    "ext_bubblepos",
    "ext_bubblespeed",
    "ext_corepos",
    "ext_dwpos",
    "ext_dwspeed",
    "ext_dwtilt",
    "ext_dwxpos",
    "ext_topologicalcharge",
    "ext_topologicalchargedensity",
    "ext_topologicalchargedensitylattice",
    "ext_topologicalchargelattice"
  ],
  "Spin-transfer Torque": ["xi", "STTorque"],
  Strain: ["exx", "exy", "exz", "eyy", "eyz", "ezz"],
  Current: ["J", "Pol"],
  Slonczewski: ["EpsilonPrime", "FixedLayer", "FreeLayerThickness", "Lambda"],
  "Magneto-elastic-constants": ["B1", "B2", "C11", "C12", "C44", "eta", "rho"],
  "Magneto-elastic-dynamics": ["F_mel", "u", "du", "normStrain", "normStress", "shearStrain", "shearStress", "force_density", "poynting"],
  Miscellaneous: [
    "frozenspins",
    "NoDemagSpins",
    "MFM",
    "spinAngle",
    "LLtorque",
    "m_full",
    "Temp",
    "geom"
  ]
};
const dynQuantities = writable({});
const dynQuantitiesCat = writable([]);
function QuantityDropdown($$payload, $$props) {
  push();
  var $$store_subs;
  let dropdownOpen = false;
  let $$settled = true;
  let $$inner_payload;
  function $$render_inner($$payload2) {
    Button($$payload2, {
      outline: true,
      class: "h-11 w-full justify-between",
      children: ($$payload3) => {
        $$payload3.out += `<span>Quantity:</span> <span class="truncate font-bold text-white">${escape_html(store_get($$store_subs ??= {}, "$p", previewState).quantity)}</span> `;
        ChevronDownOutline($$payload3, {
          class: "ms-2 h-6 w-6 text-white dark:text-white"
        });
        $$payload3.out += `<!---->`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----> `;
    Dropdown($$payload2, {
      get open() {
        return dropdownOpen;
      },
      set open($$value) {
        dropdownOpen = $$value;
        $$settled = false;
      },
      placement: "right-start",
      children: ($$payload3) => {
        const each_array = ensure_array_like(quantities["Common"]);
        const each_array_1 = ensure_array_like(Object.entries(quantities));
        const each_array_3 = ensure_array_like(store_get($$store_subs ??= {}, "$dynQuantitiesCat", dynQuantitiesCat));
        $$payload3.out += `<!--[-->`;
        for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
          let quantity = each_array[$$index];
          DropdownItem($$payload3, {
            children: ($$payload4) => {
              $$payload4.out += `<!---->${escape_html(quantity)}`;
            },
            $$slots: { default: true }
          });
        }
        $$payload3.out += `<!--]--> `;
        DropdownDivider($$payload3, {});
        $$payload3.out += `<!----> <!--[-->`;
        for (let $$index_2 = 0, $$length = each_array_1.length; $$index_2 < $$length; $$index_2++) {
          let [category, items] = each_array_1[$$index_2];
          if (category != "Common") {
            $$payload3.out += "<!--[-->";
            DropdownItem($$payload3, {
              class: "flex items-center justify-between",
              children: ($$payload4) => {
                $$payload4.out += `<!---->${escape_html(category)}`;
                ChevronRightOutline($$payload4, {
                  class: "text-primary-700 ms-2 h-6 w-6 dark:text-white"
                });
                $$payload4.out += `<!---->`;
              },
              $$slots: { default: true }
            });
            $$payload3.out += `<!----> `;
            Dropdown($$payload3, {
              placement: "right-start",
              trigger: "hover",
              children: ($$payload4) => {
                const each_array_2 = ensure_array_like(items);
                $$payload4.out += `<!--[-->`;
                for (let $$index_1 = 0, $$length2 = each_array_2.length; $$index_1 < $$length2; $$index_1++) {
                  let quantity = each_array_2[$$index_1];
                  DropdownItem($$payload4, {
                    children: ($$payload5) => {
                      $$payload5.out += `<!---->${escape_html(quantity)}`;
                    },
                    $$slots: { default: true }
                  });
                }
                $$payload4.out += `<!--]-->`;
              },
              $$slots: { default: true }
            });
            $$payload3.out += `<!---->`;
          } else {
            $$payload3.out += "<!--[!-->";
          }
          $$payload3.out += `<!--]-->`;
        }
        $$payload3.out += `<!--]--> <!--[-->`;
        for (let $$index_4 = 0, $$length = each_array_3.length; $$index_4 < $$length; $$index_4++) {
          let category = each_array_3[$$index_4];
          DropdownItem($$payload3, {
            class: "flex items-center justify-between",
            children: ($$payload4) => {
              $$payload4.out += `<!---->${escape_html(category)}`;
              ChevronRightOutline($$payload4, {
                class: "text-primary-700 ms-2 h-6 w-6 dark:text-white"
              });
              $$payload4.out += `<!---->`;
            },
            $$slots: { default: true }
          });
          $$payload3.out += `<!----> `;
          Dropdown($$payload3, {
            placement: "right-start",
            trigger: "hover",
            children: ($$payload4) => {
              const each_array_4 = ensure_array_like(store_get($$store_subs ??= {}, "$dynQuantities", dynQuantities)[category]);
              $$payload4.out += `<!--[-->`;
              for (let $$index_3 = 0, $$length2 = each_array_4.length; $$index_3 < $$length2; $$index_3++) {
                let quantity = each_array_4[$$index_3];
                DropdownItem($$payload4, {
                  children: ($$payload5) => {
                    $$payload5.out += `<!---->${escape_html(quantity)}`;
                  },
                  $$slots: { default: true }
                });
              }
              $$payload4.out += `<!--]-->`;
            },
            $$slots: { default: true }
          });
          $$payload3.out += `<!---->`;
        }
        $$payload3.out += `<!--]-->`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!---->`;
  }
  do {
    $$settled = true;
    $$inner_payload = copy_payload($$payload);
    $$render_inner($$inner_payload);
  } while (!$$settled);
  assign_payload($$payload, $$inner_payload);
  if ($$store_subs) unsubscribe_stores($$store_subs);
  pop();
}
function Component($$payload, $$props) {
  push();
  var $$store_subs;
  let isDisabled;
  isDisabled = store_get($$store_subs ??= {}, "$p", previewState).nComp == 1;
  let $$settled = true;
  let $$inner_payload;
  function $$render_inner($$payload2) {
    const each_array = ensure_array_like(["3D", "x", "y", "z"]);
    $$payload2.out += `<ul class="divide-x divide-gray-600 rounded-lg border border-gray-600 bg-gray-800 sm:flex"><!--[-->`;
    for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
      let choice = each_array[$$index];
      $$payload2.out += `<li class="w-full">`;
      Radio($$payload2, {
        class: `p-3 ${stringify(isDisabled ? "cursor-not-allowed opacity-50" : "")}`,
        get group() {
          return store_get($$store_subs ??= {}, "$p", previewState).component;
        },
        set group($$value) {
          store_mutate($$store_subs ??= {}, "$p", previewState, store_get($$store_subs ??= {}, "$p", previewState).component = $$value);
          $$settled = false;
        },
        value: choice,
        disabled: isDisabled,
        children: ($$payload3) => {
          $$payload3.out += `<!---->${escape_html(choice)}`;
        },
        $$slots: { default: true }
      });
      $$payload2.out += `<!----></li>`;
    }
    $$payload2.out += `<!--]--></ul>`;
  }
  do {
    $$settled = true;
    $$inner_payload = copy_payload($$payload);
    $$render_inner($$inner_payload);
  } while (!$$settled);
  assign_payload($$payload, $$inner_payload);
  if ($$store_subs) unsubscribe_stores($$store_subs);
  pop();
}
function Slider($$payload, $$props) {
  push();
  let sliderMin, sliderMax;
  let label = fallback($$props["label"], "Label");
  let values = fallback($$props["values"], () => [], true);
  let value = $$props["value"];
  let onChangeFunction = $$props["onChangeFunction"];
  let isDisabled = fallback($$props["isDisabled"], false);
  let index = values.indexOf(value);
  sliderMin = 0;
  sliderMax = values.length - 1;
  {
    const idx = values.indexOf(value);
    if (idx !== -1) {
      index = idx;
    } else {
      index = 0;
      value = values[0];
    }
  }
  if (values[index] !== value) {
    value = values[index];
  }
  $$payload.out += `<div${attr("class", `relative h-11 overflow-hidden rounded-md border border-gray-600 bg-gray-800 ${stringify(isDisabled ? "cursor-not-allowed opacity-50" : "")}`)}${attr("style", `--index:${stringify(index)}; --min:${stringify(sliderMin)}; --max:${stringify(sliderMax)};`)}><div class="filled-portion absolute left-0 top-0 h-full rounded-md bg-gray-700 svelte-gvofwu"></div> <input type="range"${attr("min", sliderMin)}${attr("max", sliderMax)} step="1"${attr("value", index)} class="absolute left-0 top-0 h-full w-full cursor-pointer opacity-0"${attr("disabled", isDisabled, true)}> <div class="pointer-events-none absolute inset-0 flex items-center justify-between px-4"><span class="text-gray-400">${escape_html(values[0])}</span> <span class="text-gray-400">${escape_html(values[values.length - 1])}</span></div> <div class="pointer-events-none absolute inset-0 flex items-center justify-center"><span class="text-gray-200">${escape_html(label)} = ${escape_html(value)}</span></div></div>`;
  bind_props($$props, {
    label,
    values,
    value,
    onChangeFunction,
    isDisabled
  });
  pop();
}
function Layer($$payload, $$props) {
  push();
  var $$store_subs;
  let values, layer;
  let isDisabled;
  isDisabled = store_get($$store_subs ??= {}, "$meshState", meshState).Nz < 2;
  values = Array.from(
    {
      length: store_get($$store_subs ??= {}, "$meshState", meshState).Nz
    },
    (_, i) => i
  );
  layer = store_get($$store_subs ??= {}, "$p", previewState).layer;
  let $$settled = true;
  let $$inner_payload;
  function $$render_inner($$payload2) {
    Slider($$payload2, {
      label: "Z Layer",
      get value() {
        return layer;
      },
      set value($$value) {
        layer = $$value;
        $$settled = false;
      },
      values,
      onChangeFunction: postLayer,
      isDisabled
    });
  }
  do {
    $$settled = true;
    $$inner_payload = copy_payload($$payload);
    $$render_inner($$inner_payload);
  } while (!$$settled);
  assign_payload($$payload, $$inner_payload);
  if ($$store_subs) unsubscribe_stores($$store_subs);
  pop();
}
function XDataPoints($$payload, $$props) {
  push();
  var $$store_subs;
  let $$settled = true;
  let $$inner_payload;
  function $$render_inner($$payload2) {
    Slider($$payload2, {
      label: "X Data Points",
      get value() {
        return store_get($$store_subs ??= {}, "$p", previewState).xChosenSize;
      },
      set value($$value) {
        store_mutate($$store_subs ??= {}, "$p", previewState, store_get($$store_subs ??= {}, "$p", previewState).xChosenSize = $$value);
        $$settled = false;
      },
      values: store_get($$store_subs ??= {}, "$p", previewState).xPossibleSizes,
      onChangeFunction: postXChosenSize
    });
  }
  do {
    $$settled = true;
    $$inner_payload = copy_payload($$payload);
    $$render_inner($$inner_payload);
  } while (!$$settled);
  assign_payload($$payload, $$inner_payload);
  if ($$store_subs) unsubscribe_stores($$store_subs);
  pop();
}
function YDataPoints($$payload, $$props) {
  push();
  var $$store_subs;
  let $$settled = true;
  let $$inner_payload;
  function $$render_inner($$payload2) {
    Slider($$payload2, {
      label: "Y Data Points",
      get value() {
        return store_get($$store_subs ??= {}, "$p", previewState).yChosenSize;
      },
      set value($$value) {
        store_mutate($$store_subs ??= {}, "$p", previewState, store_get($$store_subs ??= {}, "$p", previewState).yChosenSize = $$value);
        $$settled = false;
      },
      values: store_get($$store_subs ??= {}, "$p", previewState).yPossibleSizes,
      onChangeFunction: postYChosenSize
    });
  }
  do {
    $$settled = true;
    $$inner_payload = copy_payload($$payload);
    $$render_inner($$inner_payload);
  } while (!$$settled);
  assign_payload($$payload, $$inner_payload);
  if ($$store_subs) unsubscribe_stores($$store_subs);
  pop();
}
function ResetCamera($$payload, $$props) {
  push();
  var $$store_subs;
  let isDisabled;
  isDisabled = store_get($$store_subs ??= {}, "$p", previewState).nComp != 3;
  Button($$payload, {
    outline: true,
    class: `h-11 w-full ${stringify(isDisabled ? "cursor-not-allowed opacity-50" : "")}`,
    disabled: isDisabled,
    children: ($$payload2) => {
      $$payload2.out += `<!---->Reset Camera`;
    },
    $$slots: { default: true }
  });
  if ($$store_subs) unsubscribe_stores($$store_subs);
  pop();
}
function Preview($$payload, $$props) {
  push();
  var $$store_subs;
  $$payload.out += `<section class="svelte-14whbff"><h2 class="mb-4 text-2xl font-semibold">Preview</h2> <div class="m-1 flex flex-wrap svelte-14whbff" id="parent-fields"><div class="basis-1/2 svelte-14whbff">`;
  QuantityDropdown($$payload);
  $$payload.out += `<!----></div> <div class="basis-1/2 svelte-14whbff">`;
  Component($$payload);
  $$payload.out += `<!----></div> <div class="basis-1/2 svelte-14whbff">`;
  if (store_get($$store_subs ??= {}, "$p", previewState).xPossibleSizes.length > 0) {
    $$payload.out += "<!--[-->";
    XDataPoints($$payload);
  } else {
    $$payload.out += "<!--[!-->";
  }
  $$payload.out += `<!--]--></div> <div class="basis-1/2 svelte-14whbff">`;
  if (store_get($$store_subs ??= {}, "$p", previewState).yPossibleSizes.length > 0) {
    $$payload.out += "<!--[-->";
    YDataPoints($$payload);
  } else {
    $$payload.out += "<!--[!-->";
  }
  $$payload.out += `<!--]--></div> <div class="basis-1/2 svelte-14whbff">`;
  ResetCamera($$payload);
  $$payload.out += `<!----></div> <div class="basis-1/2 svelte-14whbff">`;
  Layer($$payload);
  $$payload.out += `<!----></div></div> <hr> <div class="relative h-[500px] w-full">`;
  if (store_get($$store_subs ??= {}, "$p", previewState).scalarField == null && store_get($$store_subs ??= {}, "$p", previewState).vectorFieldPositions == null) {
    $$payload.out += "<!--[-->";
    $$payload.out += `<div class="absolute inset-0 flex items-center justify-center text-6xl text-gray-600">NO DATA</div>`;
  } else {
    $$payload.out += "<!--[!-->";
  }
  $$payload.out += `<!--]--> <div id="container" class="svelte-14whbff"></div></div> <hr></section>`;
  if ($$store_subs) unsubscribe_stores($$store_subs);
  pop();
}
function MaxPoints($$payload, $$props) {
  push();
  var $$store_subs;
  let value = "";
  let $$settled = true;
  let $$inner_payload;
  function $$render_inner($$payload2) {
    ButtonGroup($$payload2, {
      class: "flex h-11",
      children: ($$payload3) => {
        InputAddon($$payload3, {
          class: "w-fit whitespace-nowrap !bg-transparent ",
          children: ($$payload4) => {
            $$payload4.out += `<!---->Max Points`;
          },
          $$slots: { default: true }
        });
        $$payload3.out += `<!----> `;
        Input($$payload3, {
          class: "w-full truncate",
          get value() {
            return value;
          },
          set value($$value) {
            value = $$value;
            $$settled = false;
          },
          placeholder: ` ${stringify(store_get($$store_subs ??= {}, "$tablePlotState", tablePlotState).maxPoints)}`
        });
        $$payload3.out += `<!---->`;
      },
      $$slots: { default: true }
    });
  }
  do {
    $$settled = true;
    $$inner_payload = copy_payload($$payload);
    $$render_inner($$inner_payload);
  } while (!$$settled);
  assign_payload($$payload, $$inner_payload);
  if ($$store_subs) unsubscribe_stores($$store_subs);
  pop();
}
function AutoSaveInterval($$payload, $$props) {
  push();
  var $$store_subs;
  let value = "";
  let $$settled = true;
  let $$inner_payload;
  function $$render_inner($$payload2) {
    ButtonGroup($$payload2, {
      class: "flex h-11",
      children: ($$payload3) => {
        InputAddon($$payload3, {
          class: "w-fit whitespace-nowrap !bg-transparent",
          children: ($$payload4) => {
            $$payload4.out += `<!---->AutoSave Interval`;
          },
          $$slots: { default: true }
        });
        $$payload3.out += `<!----> `;
        Input($$payload3, {
          class: "w-full truncate",
          get value() {
            return value;
          },
          set value($$value) {
            value = $$value;
            $$settled = false;
          },
          placeholder: ` ${stringify(store_get($$store_subs ??= {}, "$tablePlotState", tablePlotState).autoSaveInterval)}`
        });
        $$payload3.out += `<!----> `;
        InputAddon($$payload3, {
          class: "w-16 !bg-transparent",
          children: ($$payload4) => {
            $$payload4.out += `<!---->s`;
          },
          $$slots: { default: true }
        });
        $$payload3.out += `<!---->`;
      },
      $$slots: { default: true }
    });
  }
  do {
    $$settled = true;
    $$inner_payload = copy_payload($$payload);
    $$render_inner($$inner_payload);
  } while (!$$settled);
  assign_payload($$payload, $$inner_payload);
  if ($$store_subs) unsubscribe_stores($$store_subs);
  pop();
}
function XColumn($$payload, $$props) {
  push();
  var $$store_subs;
  let dropdownOpen = false;
  let $$settled = true;
  let $$inner_payload;
  function $$render_inner($$payload2) {
    Button($$payload2, {
      outline: true,
      color: "primary",
      class: "h-11 w-full justify-between",
      children: ($$payload3) => {
        $$payload3.out += `<span>X Axis:</span> <span class="truncate font-bold text-white">${escape_html(store_get($$store_subs ??= {}, "$tablePlotState", tablePlotState).xColumn)}</span> `;
        ChevronDownOutline($$payload3, { class: "h-5 w-5 text-gray-500" });
        $$payload3.out += `<!---->`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----> `;
    Dropdown($$payload2, {
      get open() {
        return dropdownOpen;
      },
      set open($$value) {
        dropdownOpen = $$value;
        $$settled = false;
      },
      class: "w-3/4",
      children: ($$payload3) => {
        const each_array = ensure_array_like(store_get($$store_subs ??= {}, "$tablePlotState", tablePlotState).columns);
        $$payload3.out += `<!--[-->`;
        for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
          let q = each_array[$$index];
          DropdownItem($$payload3, {
            children: ($$payload4) => {
              $$payload4.out += `<!---->${escape_html(q)}`;
            },
            $$slots: { default: true }
          });
        }
        $$payload3.out += `<!--]-->`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!---->`;
  }
  do {
    $$settled = true;
    $$inner_payload = copy_payload($$payload);
    $$render_inner($$inner_payload);
  } while (!$$settled);
  assign_payload($$payload, $$inner_payload);
  if ($$store_subs) unsubscribe_stores($$store_subs);
  pop();
}
function YColumn($$payload, $$props) {
  push();
  var $$store_subs;
  let dropdownOpen = false;
  let $$settled = true;
  let $$inner_payload;
  function $$render_inner($$payload2) {
    Button($$payload2, {
      outline: true,
      color: "primary",
      class: "h-11 w-full justify-between",
      children: ($$payload3) => {
        $$payload3.out += `<span>Y Axis:</span> <span class="truncate font-bold text-white">${escape_html(store_get($$store_subs ??= {}, "$tablePlotState", tablePlotState).yColumn)}</span> `;
        ChevronDownOutline($$payload3, { class: "h-5 w-5 text-gray-500" });
        $$payload3.out += `<!---->`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----> `;
    Dropdown($$payload2, {
      get open() {
        return dropdownOpen;
      },
      set open($$value) {
        dropdownOpen = $$value;
        $$settled = false;
      },
      class: "w-3/4",
      children: ($$payload3) => {
        const each_array = ensure_array_like(store_get($$store_subs ??= {}, "$tablePlotState", tablePlotState).columns);
        $$payload3.out += `<!--[-->`;
        for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
          let q = each_array[$$index];
          DropdownItem($$payload3, {
            children: ($$payload4) => {
              $$payload4.out += `<!---->${escape_html(q)}`;
            },
            $$slots: { default: true }
          });
        }
        $$payload3.out += `<!--]-->`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!---->`;
  }
  do {
    $$settled = true;
    $$inner_payload = copy_payload($$payload);
    $$render_inner($$inner_payload);
  } while (!$$settled);
  assign_payload($$payload, $$inner_payload);
  if ($$store_subs) unsubscribe_stores($$store_subs);
  pop();
}
function Step($$payload, $$props) {
  push();
  var $$store_subs;
  let value = "";
  let $$settled = true;
  let $$inner_payload;
  function $$render_inner($$payload2) {
    ButtonGroup($$payload2, {
      class: "flex h-11",
      children: ($$payload3) => {
        InputAddon($$payload3, {
          class: "w-16 whitespace-nowrap !bg-transparent ",
          children: ($$payload4) => {
            $$payload4.out += `<!---->Step`;
          },
          $$slots: { default: true }
        });
        $$payload3.out += `<!----> `;
        Input($$payload3, {
          class: "w-full truncate",
          get value() {
            return value;
          },
          set value($$value) {
            value = $$value;
            $$settled = false;
          },
          placeholder: ` ${stringify(store_get($$store_subs ??= {}, "$tablePlotState", tablePlotState).step)}`
        });
        $$payload3.out += `<!---->`;
      },
      $$slots: { default: true }
    });
  }
  do {
    $$settled = true;
    $$inner_payload = copy_payload($$payload);
    $$render_inner($$inner_payload);
  } while (!$$settled);
  assign_payload($$payload, $$inner_payload);
  if ($$store_subs) unsubscribe_stores($$store_subs);
  pop();
}
function TablePlot($$payload, $$props) {
  push();
  var $$store_subs;
  $$payload.out += `<section class="svelte-1wmy1p7"><h2 class="mb-4 text-2xl font-semibold">Table Plot</h2> `;
  if (store_get($$store_subs ??= {}, "$tablePlotState", tablePlotState).data.length === 0) {
    $$payload.out += "<!--[-->";
    $$payload.out += `<div class="msg svelte-1wmy1p7"><p>No table data, use <code>TableSave()</code> or set a non-zero autosave interval to save data.</p> <div class="mt-5 w-72">`;
    AutoSaveInterval($$payload);
    $$payload.out += `<!----></div></div>`;
  } else {
    $$payload.out += "<!--[!-->";
    $$payload.out += `<div class="m-1 flex flex-wrap svelte-1wmy1p7" id="parent-fields"><div class="basis-1/2 svelte-1wmy1p7">`;
    XColumn($$payload);
    $$payload.out += `<!----></div> <div class="basis-1/2 svelte-1wmy1p7">`;
    YColumn($$payload);
    $$payload.out += `<!----></div> <div class="basis-4/12 svelte-1wmy1p7">`;
    MaxPoints($$payload);
    $$payload.out += `<!----></div> <div class="basis-2/12 svelte-1wmy1p7">`;
    Step($$payload);
    $$payload.out += `<!----></div> <div class="basis-6/12 svelte-1wmy1p7">`;
    AutoSaveInterval($$payload);
    $$payload.out += `<!----></div></div> <hr> <div id="table-plot" class="svelte-1wmy1p7"></div>`;
  }
  $$payload.out += `<!--]--></section>`;
  if ($$store_subs) unsubscribe_stores($$store_subs);
  pop();
}
function Solver($$payload, $$props) {
  push();
  var $$store_subs;
  let solvertypes = [
    "bw_euler",
    "euler",
    "heun",
    "rk4",
    "rk23",
    "rk45",
    "rkf56",
    "elasRK4",
    "magelasRK4",
    "elasLF",
    "elasYOSH",
    "magelasRK4_vary_time"
  ];
  let runSeconds = "1e-9";
  let runSteps = "100";
  let dropdownOpen = false;
  let $$settled = true;
  let $$inner_payload;
  function $$render_inner($$payload2) {
    $$payload2.out += `<section class="svelte-1ika8pl"><h2 class="mb-6 text-2xl font-semibold">Solver</h2> <div class="grid grid-cols-2 gap-x-3 gap-y-6"><div class="space-y-4"><div>`;
    Button($$payload2, {
      outline: true,
      color: "primary",
      class: "flex w-full items-center justify-between",
      children: ($$payload3) => {
        $$payload3.out += `<span>Solver:</span> <span class="truncate font-bold text-white">${escape_html(store_get($$store_subs ??= {}, "$solverState", solverState).type)}</span> `;
        ChevronDownOutline($$payload3, { class: "h-5 w-5 text-gray-500" });
        $$payload3.out += `<!---->`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----> `;
    Dropdown($$payload2, {
      get open() {
        return dropdownOpen;
      },
      set open($$value) {
        dropdownOpen = $$value;
        $$settled = false;
      },
      class: "w-56",
      placement: "bottom-start",
      children: ($$payload3) => {
        const each_array = ensure_array_like(solvertypes);
        $$payload3.out += `<!--[-->`;
        for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
          let solvertype = each_array[$$index];
          DropdownItem($$payload3, {
            class: "flex items-center gap-2 text-base font-semibold",
            children: ($$payload4) => {
              $$payload4.out += `<!---->${escape_html(solvertype)}`;
            },
            $$slots: { default: true }
          });
        }
        $$payload3.out += `<!--]-->`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----></div> <div class="field svelte-1ika8pl">`;
    Button($$payload2, {
      outline: true,
      children: ($$payload3) => {
        $$payload3.out += `<!---->Run`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----> `;
    ButtonGroup($$payload2, {
      class: "h-11 w-full pl-3",
      children: ($$payload3) => {
        Input($$payload3, {
          class: "w-full",
          id: "run_seconds",
          get value() {
            return runSeconds;
          },
          set value($$value) {
            runSeconds = $$value;
            $$settled = false;
          },
          placeholder: "Time in seconds"
        });
        $$payload3.out += `<!----> `;
        InputAddon($$payload3, {
          class: "w-16 !bg-transparent",
          children: ($$payload4) => {
            $$payload4.out += `<!---->s`;
          },
          $$slots: { default: true }
        });
        $$payload3.out += `<!---->`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----></div> <div class="field svelte-1ika8pl">`;
    Button($$payload2, {
      class: "inline-flex h-11 items-center whitespace-nowrap px-4 py-2",
      outline: true,
      children: ($$payload3) => {
        $$payload3.out += `<!---->Run Steps`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----> <div class="w-full pl-3">`;
    Input($$payload2, {
      id: "run_steps",
      get value() {
        return runSteps;
      },
      set value($$value) {
        runSteps = $$value;
        $$settled = false;
      },
      placeholder: "Number of steps"
    });
    $$payload2.out += `<!----></div></div> <div>`;
    Button($$payload2, {
      outline: true,
      class: "w-full",
      children: ($$payload3) => {
        $$payload3.out += `<!---->Relax`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----></div> <div>`;
    Button($$payload2, {
      outline: true,
      class: "w-full",
      children: ($$payload3) => {
        $$payload3.out += `<!---->Minimize`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----></div> <div>`;
    Button($$payload2, {
      outline: true,
      class: "w-full",
      children: ($$payload3) => {
        $$payload3.out += `<!---->Break`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----></div></div> <div class="space-y-4"><div class="field svelte-1ika8pl">`;
    ButtonGroup($$payload2, {
      class: "btn-group",
      children: ($$payload3) => {
        InputAddon($$payload3, {
          class: "w-44 !bg-transparent",
          children: ($$payload4) => {
            $$payload4.out += `<!---->Steps`;
          },
          $$slots: { default: true }
        });
        $$payload3.out += `<!----> `;
        Input($$payload3, {
          class: "w-full",
          value: store_get($$store_subs ??= {}, "$solverState", solverState).steps,
          readonly: true
        });
        $$payload3.out += `<!----> `;
        InputAddon($$payload3, { class: "w-24 !bg-transparent" });
        $$payload3.out += `<!---->`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----></div> <div class="field svelte-1ika8pl">`;
    ButtonGroup($$payload2, {
      class: "btn-group",
      children: ($$payload3) => {
        InputAddon($$payload3, {
          class: "w-44 !bg-transparent",
          children: ($$payload4) => {
            $$payload4.out += `<!---->Time`;
          },
          $$slots: { default: true }
        });
        $$payload3.out += `<!----> `;
        Input($$payload3, {
          class: "w-full",
          type: "text",
          value: store_get($$store_subs ??= {}, "$solverState", solverState).time.toExponential(3),
          readonly: true
        });
        $$payload3.out += `<!----> `;
        InputAddon($$payload3, {
          class: "w-24 !bg-transparent",
          children: ($$payload4) => {
            $$payload4.out += `<!---->s`;
          },
          $$slots: { default: true }
        });
        $$payload3.out += `<!---->`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----></div> <div class="field svelte-1ika8pl">`;
    ButtonGroup($$payload2, {
      class: "btn-group",
      children: ($$payload3) => {
        InputAddon($$payload3, {
          class: "w-44 !bg-transparent",
          children: ($$payload4) => {
            $$payload4.out += `<!---->dt`;
          },
          $$slots: { default: true }
        });
        $$payload3.out += `<!----> `;
        Input($$payload3, {
          class: "w-full",
          type: "text",
          value: store_get($$store_subs ??= {}, "$solverState", solverState).dt.toExponential(3),
          readonly: true
        });
        $$payload3.out += `<!----> `;
        InputAddon($$payload3, {
          class: "w-24 !bg-transparent",
          children: ($$payload4) => {
            $$payload4.out += `<!---->s`;
          },
          $$slots: { default: true }
        });
        $$payload3.out += `<!---->`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----></div> <div class="field svelte-1ika8pl">`;
    ButtonGroup($$payload2, {
      class: "btn-group",
      children: ($$payload3) => {
        InputAddon($$payload3, {
          class: "w-44 !bg-transparent",
          children: ($$payload4) => {
            $$payload4.out += `<!---->Err/step`;
          },
          $$slots: { default: true }
        });
        $$payload3.out += `<!----> `;
        Input($$payload3, {
          class: "w-full",
          type: "text",
          value: store_get($$store_subs ??= {}, "$solverState", solverState).errPerStep.toExponential(3),
          readonly: true
        });
        $$payload3.out += `<!----> `;
        InputAddon($$payload3, { class: "w-24 !bg-transparent" });
        $$payload3.out += `<!---->`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----></div> <div class="field svelte-1ika8pl">`;
    ButtonGroup($$payload2, {
      class: "btn-group",
      children: ($$payload3) => {
        InputAddon($$payload3, {
          class: "w-44 !bg-transparent",
          children: ($$payload4) => {
            $$payload4.out += `<!---->Max Torque`;
          },
          $$slots: { default: true }
        });
        $$payload3.out += `<!----> `;
        Input($$payload3, {
          class: "w-full",
          type: "text",
          value: store_get($$store_subs ??= {}, "$solverState", solverState).maxTorque.toExponential(3),
          readonly: true
        });
        $$payload3.out += `<!----> `;
        InputAddon($$payload3, {
          class: "w-24 !bg-transparent",
          children: ($$payload4) => {
            $$payload4.out += `<!---->T`;
          },
          $$slots: { default: true }
        });
        $$payload3.out += `<!---->`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----></div> <div class="field svelte-1ika8pl">`;
    ButtonGroup($$payload2, {
      class: "btn-group",
      children: ($$payload3) => {
        InputAddon($$payload3, {
          class: "w-44 !bg-transparent",
          children: ($$payload4) => {
            $$payload4.out += `<!---->Fixdt`;
          },
          $$slots: { default: true }
        });
        $$payload3.out += `<!----> `;
        Input($$payload3, {
          class: "w-full",
          type: "number",
          get value() {
            return store_get($$store_subs ??= {}, "$solverState", solverState).fixdt;
          },
          set value($$value) {
            store_mutate($$store_subs ??= {}, "$solverState", solverState, store_get($$store_subs ??= {}, "$solverState", solverState).fixdt = $$value);
            $$settled = false;
          }
        });
        $$payload3.out += `<!----> `;
        InputAddon($$payload3, {
          class: "w-24 !bg-transparent",
          children: ($$payload4) => {
            $$payload4.out += `<!---->s`;
          },
          $$slots: { default: true }
        });
        $$payload3.out += `<!---->`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----></div> <div class="field svelte-1ika8pl">`;
    ButtonGroup($$payload2, {
      class: "btn-group",
      children: ($$payload3) => {
        InputAddon($$payload3, {
          class: "w-44 !bg-transparent",
          children: ($$payload4) => {
            $$payload4.out += `<!---->Mindt`;
          },
          $$slots: { default: true }
        });
        $$payload3.out += `<!----> `;
        Input($$payload3, {
          class: "w-full",
          type: "number",
          get value() {
            return store_get($$store_subs ??= {}, "$solverState", solverState).mindt;
          },
          set value($$value) {
            store_mutate($$store_subs ??= {}, "$solverState", solverState, store_get($$store_subs ??= {}, "$solverState", solverState).mindt = $$value);
            $$settled = false;
          }
        });
        $$payload3.out += `<!----> `;
        InputAddon($$payload3, {
          class: "w-24 !bg-transparent",
          children: ($$payload4) => {
            $$payload4.out += `<!---->s`;
          },
          $$slots: { default: true }
        });
        $$payload3.out += `<!---->`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----></div> <div class="field svelte-1ika8pl">`;
    ButtonGroup($$payload2, {
      class: "btn-group",
      children: ($$payload3) => {
        InputAddon($$payload3, {
          class: "w-44 !bg-transparent",
          children: ($$payload4) => {
            $$payload4.out += `<!---->Maxdt`;
          },
          $$slots: { default: true }
        });
        $$payload3.out += `<!----> `;
        Input($$payload3, {
          class: "w-full",
          type: "number",
          get value() {
            return store_get($$store_subs ??= {}, "$solverState", solverState).maxdt;
          },
          set value($$value) {
            store_mutate($$store_subs ??= {}, "$solverState", solverState, store_get($$store_subs ??= {}, "$solverState", solverState).maxdt = $$value);
            $$settled = false;
          }
        });
        $$payload3.out += `<!----> `;
        InputAddon($$payload3, {
          class: "w-24 !bg-transparent",
          children: ($$payload4) => {
            $$payload4.out += `<!---->s`;
          },
          $$slots: { default: true }
        });
        $$payload3.out += `<!---->`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----></div> <div class="field svelte-1ika8pl"><div class="btn-group svelte-1ika8pl">`;
    ButtonGroup($$payload2, {
      children: ($$payload3) => {
        InputAddon($$payload3, {
          class: "w-44 !bg-transparent",
          children: ($$payload4) => {
            $$payload4.out += `<!---->MaxErr`;
          },
          $$slots: { default: true }
        });
        $$payload3.out += `<!----> `;
        Input($$payload3, {
          class: "w-full",
          type: "number",
          get value() {
            return store_get($$store_subs ??= {}, "$solverState", solverState).maxerr;
          },
          set value($$value) {
            store_mutate($$store_subs ??= {}, "$solverState", solverState, store_get($$store_subs ??= {}, "$solverState", solverState).maxerr = $$value);
            $$settled = false;
          }
        });
        $$payload3.out += `<!----> `;
        InputAddon($$payload3, { class: "w-24 !bg-transparent" });
        $$payload3.out += `<!---->`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----></div></div> <div class="field svelte-1ika8pl"><div class="btn-group svelte-1ika8pl">`;
    ButtonGroup($$payload2, {
      children: ($$payload3) => {
        InputAddon($$payload3, {
          class: "w-44 !bg-transparent",
          children: ($$payload4) => {
            $$payload4.out += `<!---->NUndone`;
          },
          $$slots: { default: true }
        });
        $$payload3.out += `<!----> `;
        Input($$payload3, {
          class: "w-full",
          type: "number",
          get value() {
            return store_get($$store_subs ??= {}, "$solverState", solverState).nundone;
          },
          set value($$value) {
            store_mutate($$store_subs ??= {}, "$solverState", solverState, store_get($$store_subs ??= {}, "$solverState", solverState).nundone = $$value);
            $$settled = false;
          },
          readonly: true
        });
        $$payload3.out += `<!----> `;
        InputAddon($$payload3, { class: "w-24 !bg-transparent" });
        $$payload3.out += `<!---->`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----></div></div></div></div></section>`;
  }
  do {
    $$settled = true;
    $$inner_payload = copy_payload($$payload);
    $$render_inner($$inner_payload);
  } while (!$$settled);
  assign_payload($$payload, $$inner_payload);
  if ($$store_subs) unsubscribe_stores($$store_subs);
  pop();
}
function Console($$payload, $$props) {
  push();
  var $$store_subs;
  let command = "";
  async function scrollDown() {
  }
  {
    store_get($$store_subs ??= {}, "$consoleState", consoleState).hist;
    scrollDown();
  }
  $$payload.out += `<section class="svelte-1lgunmx"><h2 class="mb-4 text-2xl font-semibold">Console</h2> <div class="flex flex-col gap-2"><div class="code svelte-1lgunmx">${html(Prism.highlight(store_get($$store_subs ??= {}, "$consoleState", consoleState).hist, Prism.languages["go"], "go"))}</div> <br> <input placeholder="type commands here, or up/down" size="86"${attr("value", command)} class="svelte-1lgunmx"></div></section>`;
  if ($$store_subs) unsubscribe_stores($$store_subs);
  pop();
}
function Mesh($$payload, $$props) {
  push();
  var $$store_subs;
  $$payload.out += `<section class="svelte-2fswe"><h2 class="mb-4 text-2xl font-semibold">Mesh</h2> <div class="mb-6 grid gap-6 md:grid-cols-3">`;
  ButtonGroup($$payload, {
    class: "h-11 w-full",
    children: ($$payload2) => {
      InputAddon($$payload2, {
        class: "w-28 !bg-transparent",
        children: ($$payload3) => {
          $$payload3.out += `<!---->dx`;
        },
        $$slots: { default: true }
      });
      $$payload2.out += `<!----> `;
      Input($$payload2, {
        type: "text",
        placeholder: store_get($$store_subs ??= {}, "$m", meshState).dx.toPrecision(8)
      });
      $$payload2.out += `<!----> `;
      InputAddon($$payload2, {
        class: "w-14 !bg-transparent",
        children: ($$payload3) => {
          $$payload3.out += `<!---->m`;
        },
        $$slots: { default: true }
      });
      $$payload2.out += `<!---->`;
    },
    $$slots: { default: true }
  });
  $$payload.out += `<!----> `;
  ButtonGroup($$payload, {
    class: "h-11 w-full",
    children: ($$payload2) => {
      InputAddon($$payload2, {
        class: "w-28 !bg-transparent",
        children: ($$payload3) => {
          $$payload3.out += `<!---->dy`;
        },
        $$slots: { default: true }
      });
      $$payload2.out += `<!----> `;
      Input($$payload2, {
        type: "text",
        placeholder: store_get($$store_subs ??= {}, "$m", meshState).dy.toPrecision(8)
      });
      $$payload2.out += `<!----> `;
      InputAddon($$payload2, {
        class: "w-14 !bg-transparent",
        children: ($$payload3) => {
          $$payload3.out += `<!---->m`;
        },
        $$slots: { default: true }
      });
      $$payload2.out += `<!---->`;
    },
    $$slots: { default: true }
  });
  $$payload.out += `<!----> `;
  ButtonGroup($$payload, {
    class: "h-11 w-full",
    children: ($$payload2) => {
      InputAddon($$payload2, {
        class: "w-28 !bg-transparent",
        children: ($$payload3) => {
          $$payload3.out += `<!---->dz`;
        },
        $$slots: { default: true }
      });
      $$payload2.out += `<!----> `;
      Input($$payload2, {
        type: "text",
        placeholder: store_get($$store_subs ??= {}, "$m", meshState).dz.toPrecision(8)
      });
      $$payload2.out += `<!----> `;
      InputAddon($$payload2, {
        class: "w-14 !bg-transparent",
        children: ($$payload3) => {
          $$payload3.out += `<!---->m`;
        },
        $$slots: { default: true }
      });
      $$payload2.out += `<!---->`;
    },
    $$slots: { default: true }
  });
  $$payload.out += `<!----></div> <div class="mb-6 grid gap-6 md:grid-cols-3">`;
  ButtonGroup($$payload, {
    class: "h-11 w-full",
    children: ($$payload2) => {
      InputAddon($$payload2, {
        class: "w-28 !bg-transparent",
        children: ($$payload3) => {
          $$payload3.out += `<!---->Nx`;
        },
        $$slots: { default: true }
      });
      $$payload2.out += `<!----> `;
      Input($$payload2, {
        type: "text",
        placeholder: ` ${stringify(store_get($$store_subs ??= {}, "$m", meshState).Nx)}`
      });
      $$payload2.out += `<!----> `;
      InputAddon($$payload2, { class: "w-14 !bg-transparent" });
      $$payload2.out += `<!---->`;
    },
    $$slots: { default: true }
  });
  $$payload.out += `<!----> `;
  ButtonGroup($$payload, {
    class: "h-11 w-full",
    children: ($$payload2) => {
      InputAddon($$payload2, {
        class: "w-28 !bg-transparent",
        children: ($$payload3) => {
          $$payload3.out += `<!---->Ny`;
        },
        $$slots: { default: true }
      });
      $$payload2.out += `<!----> `;
      Input($$payload2, {
        type: "text",
        placeholder: ` ${stringify(store_get($$store_subs ??= {}, "$m", meshState).Ny)}`
      });
      $$payload2.out += `<!----> `;
      InputAddon($$payload2, { class: "w-14 !bg-transparent" });
      $$payload2.out += `<!---->`;
    },
    $$slots: { default: true }
  });
  $$payload.out += `<!----> `;
  ButtonGroup($$payload, {
    class: "h-11 w-full",
    children: ($$payload2) => {
      InputAddon($$payload2, {
        class: "w-28 !bg-transparent",
        children: ($$payload3) => {
          $$payload3.out += `<!---->Nz`;
        },
        $$slots: { default: true }
      });
      $$payload2.out += `<!----> `;
      Input($$payload2, {
        type: "text",
        placeholder: ` ${stringify(store_get($$store_subs ??= {}, "$m", meshState).Nz)}`
      });
      $$payload2.out += `<!----> `;
      InputAddon($$payload2, { class: "w-14 !bg-transparent" });
      $$payload2.out += `<!---->`;
    },
    $$slots: { default: true }
  });
  $$payload.out += `<!----></div> <div class="mb-6 grid gap-6 md:grid-cols-3">`;
  ButtonGroup($$payload, {
    class: "h-11 w-full",
    children: ($$payload2) => {
      InputAddon($$payload2, {
        class: "w-28 !bg-transparent",
        children: ($$payload3) => {
          $$payload3.out += `<!---->Tx`;
        },
        $$slots: { default: true }
      });
      $$payload2.out += `<!----> `;
      Input($$payload2, {
        type: "text",
        placeholder: store_get($$store_subs ??= {}, "$m", meshState).Tx.toExponential(6)
      });
      $$payload2.out += `<!----> `;
      InputAddon($$payload2, {
        class: "w-14 !bg-transparent",
        children: ($$payload3) => {
          $$payload3.out += `<!---->m`;
        },
        $$slots: { default: true }
      });
      $$payload2.out += `<!---->`;
    },
    $$slots: { default: true }
  });
  $$payload.out += `<!----> `;
  ButtonGroup($$payload, {
    class: "h-11 w-full",
    children: ($$payload2) => {
      InputAddon($$payload2, {
        class: "w-28 !bg-transparent",
        children: ($$payload3) => {
          $$payload3.out += `<!---->Ty`;
        },
        $$slots: { default: true }
      });
      $$payload2.out += `<!----> `;
      Input($$payload2, {
        type: "text",
        placeholder: store_get($$store_subs ??= {}, "$m", meshState).Ty.toExponential(6)
      });
      $$payload2.out += `<!----> `;
      InputAddon($$payload2, {
        class: "w-14 !bg-transparent",
        children: ($$payload3) => {
          $$payload3.out += `<!---->m`;
        },
        $$slots: { default: true }
      });
      $$payload2.out += `<!---->`;
    },
    $$slots: { default: true }
  });
  $$payload.out += `<!----> `;
  ButtonGroup($$payload, {
    class: "h-11 w-full",
    children: ($$payload2) => {
      InputAddon($$payload2, {
        class: "w-28 !bg-transparent",
        children: ($$payload3) => {
          $$payload3.out += `<!---->Tz`;
        },
        $$slots: { default: true }
      });
      $$payload2.out += `<!----> `;
      Input($$payload2, {
        type: "text",
        placeholder: store_get($$store_subs ??= {}, "$m", meshState).Tz.toExponential(6)
      });
      $$payload2.out += `<!----> `;
      InputAddon($$payload2, {
        class: "w-14 !bg-transparent",
        children: ($$payload3) => {
          $$payload3.out += `<!---->m`;
        },
        $$slots: { default: true }
      });
      $$payload2.out += `<!---->`;
    },
    $$slots: { default: true }
  });
  $$payload.out += `<!----></div> <div class="mb-6 grid gap-6 md:grid-cols-3">`;
  ButtonGroup($$payload, {
    class: "h-11 w-full",
    children: ($$payload2) => {
      InputAddon($$payload2, {
        class: "w-28 !bg-transparent",
        children: ($$payload3) => {
          $$payload3.out += `<!---->PBCx`;
        },
        $$slots: { default: true }
      });
      $$payload2.out += `<!----> `;
      Input($$payload2, {
        type: "text",
        placeholder: ` ${stringify(store_get($$store_subs ??= {}, "$m", meshState).PBCx)}`
      });
      $$payload2.out += `<!----> `;
      InputAddon($$payload2, { class: "w-14 !bg-transparent" });
      $$payload2.out += `<!---->`;
    },
    $$slots: { default: true }
  });
  $$payload.out += `<!----> `;
  ButtonGroup($$payload, {
    class: "h-11 w-full",
    children: ($$payload2) => {
      InputAddon($$payload2, {
        class: "w-28 !bg-transparent",
        children: ($$payload3) => {
          $$payload3.out += `<!---->PBCy`;
        },
        $$slots: { default: true }
      });
      $$payload2.out += `<!----> `;
      Input($$payload2, {
        type: "text",
        placeholder: ` ${stringify(store_get($$store_subs ??= {}, "$m", meshState).PBCy)}`
      });
      $$payload2.out += `<!----> `;
      InputAddon($$payload2, { class: "w-14 !bg-transparent" });
      $$payload2.out += `<!---->`;
    },
    $$slots: { default: true }
  });
  $$payload.out += `<!----> `;
  ButtonGroup($$payload, {
    class: "h-11 w-full",
    children: ($$payload2) => {
      InputAddon($$payload2, {
        class: "w-28 !bg-transparent",
        children: ($$payload3) => {
          $$payload3.out += `<!---->PBCz`;
        },
        $$slots: { default: true }
      });
      $$payload2.out += `<!----> `;
      Input($$payload2, {
        type: "text",
        placeholder: ` ${stringify(store_get($$store_subs ??= {}, "$m", meshState).PBCz)}`
      });
      $$payload2.out += `<!----> `;
      InputAddon($$payload2, { class: "w-14 !bg-transparent" });
      $$payload2.out += `<!---->`;
    },
    $$slots: { default: true }
  });
  $$payload.out += `<!----></div></section>`;
  if ($$store_subs) unsubscribe_stores($$store_subs);
  pop();
}
function Parameters($$payload, $$props) {
  push();
  var $$store_subs;
  let dropdownOpen = false;
  let showZeroValues = false;
  let $$settled = true;
  let $$inner_payload;
  function $$render_inner($$payload2) {
    const each_array_1 = ensure_array_like(store_get($$store_subs ??= {}, "$p", parametersState).fields);
    $$payload2.out += `<section class="svelte-1eb79ky"><h2 class="mb-4 text-2xl font-semibold">Parameters</h2> <div class="m-3 flex flex-col gap-2"><div class="m-3 grid grid-cols-2 gap-2"><div class="flex items-center justify-center">`;
    Button($$payload2, {
      outline: true,
      children: ($$payload3) => {
        $$payload3.out += `<!---->Region: ${escape_html(store_get($$store_subs ??= {}, "$p", parametersState).selectedRegion)} `;
        ChevronDownOutline($$payload3, {
          class: "ms-2 h-6 w-6 text-white dark:text-white"
        });
        $$payload3.out += `<!---->`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----> `;
    Dropdown($$payload2, {
      get open() {
        return dropdownOpen;
      },
      set open($$value) {
        dropdownOpen = $$value;
        $$settled = false;
      },
      class: "h-48 w-40 overflow-y-auto py-1",
      children: ($$payload3) => {
        const each_array = ensure_array_like(store_get($$store_subs ??= {}, "$p", parametersState).regions);
        $$payload3.out += `<!--[-->`;
        for (let $$index = 0, $$length = each_array.length; $$index < $$length; $$index++) {
          let region = each_array[$$index];
          DropdownItem($$payload3, {
            children: ($$payload4) => {
              $$payload4.out += `<!---->${escape_html(region)}`;
            },
            $$slots: { default: true }
          });
        }
        $$payload3.out += `<!--]-->`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----></div> <div class="flex items-center justify-center">`;
    Toggle($$payload2, {
      get checked() {
        return showZeroValues;
      },
      set checked($$value) {
        showZeroValues = $$value;
        $$settled = false;
      },
      children: ($$payload3) => {
        $$payload3.out += `<!---->Show Unchanged Parameters`;
      },
      $$slots: { default: true }
    });
    $$payload2.out += `<!----></div></div></div> <div class="grid-container svelte-1eb79ky"><!--[-->`;
    for (let $$index_1 = 0, $$length = each_array_1.length; $$index_1 < $$length; $$index_1++) {
      let field = each_array_1[$$index_1];
      if (field.changed || showZeroValues) {
        $$payload2.out += "<!--[-->";
        $$payload2.out += `<div class="header svelte-1eb79ky">${escape_html(field.name)}:</div> <input${attr("placeholder", ` ${stringify(field.value)}`)} class="svelte-1eb79ky"> <div class="description svelte-1eb79ky">${escape_html(field.description)}</div>`;
      } else {
        $$payload2.out += "<!--[!-->";
      }
      $$payload2.out += `<!--]-->`;
    }
    $$payload2.out += `<!--]--></div></section>`;
  }
  do {
    $$settled = true;
    $$inner_payload = copy_payload($$payload);
    $$render_inner($$inner_payload);
  } while (!$$settled);
  assign_payload($$payload, $$inner_payload);
  if ($$store_subs) unsubscribe_stores($$store_subs);
  pop();
}
function Metrics($$payload, $$props) {
  push();
  var $$store_subs;
  $$payload.out += `<section class="svelte-wp1otw"><h2 class="mb-4 text-2xl font-semibold">Metrics</h2> `;
  if (store_get($$store_subs ??= {}, "$m", metricsState).error) {
    $$payload.out += "<!--[-->";
    $$payload.out += `<p class="text-red-500">Got an error while collecting system metrics: ${escape_html(store_get($$store_subs ??= {}, "$m", metricsState).error)}</p> `;
    Button($$payload, {
      color: "red",
      class: "mt-4",
      outline: true,
      children: ($$payload2) => {
        $$payload2.out += `<!---->Retry`;
      },
      $$slots: { default: true }
    });
    $$payload.out += `<!---->`;
  } else {
    $$payload.out += "<!--[!-->";
    $$payload.out += `<div class="grid gap-6 md:grid-cols-2">`;
    Card($$payload, {
      class: "p-6",
      children: ($$payload2) => {
        $$payload2.out += `<h3 class="mb-4 text-xl font-semibold">System Metrics</h3> <div class="space-y-6">`;
        Card($$payload2, {
          class: "p-4",
          children: ($$payload3) => {
            $$payload3.out += `<h4 class="mb-2 text-lg font-semibold">Global</h4> <ul class="space-y-4"><li><div class="flex items-center justify-between"><span class="font-medium">CPU Usage:</span> <span>${escape_html(store_get($$store_subs ??= {}, "$m", metricsState).cpuPercentTotal.toFixed(2))}%</span></div> `;
            Progressbar($$payload3, {
              animate: true,
              progress: store_get($$store_subs ??= {}, "$m", metricsState).cpuPercentTotal,
              color: "blue"
            });
            $$payload3.out += `<!----></li> <li><div class="flex items-center justify-between"><span class="font-medium">RAM Usage:</span> <span>${escape_html(store_get($$store_subs ??= {}, "$m", metricsState).ramPercentTotal.toFixed(2))}%</span></div> `;
            Progressbar($$payload3, {
              animate: true,
              progress: store_get($$store_subs ??= {}, "$m", metricsState).ramPercentTotal,
              color: "green"
            });
            $$payload3.out += `<!----></li></ul>`;
          },
          $$slots: { default: true }
        });
        $$payload2.out += `<!----> `;
        Card($$payload2, {
          class: "p-4",
          children: ($$payload3) => {
            $$payload3.out += `<h4 class="mb-2 text-lg font-semibold">Simulation</h4> <ul class="space-y-4"><li><span class="font-medium">PID:</span> ${escape_html(store_get($$store_subs ??= {}, "$m", metricsState).pid)}</li> <li><div class="flex items-center justify-between"><span class="font-medium">CPU Usage:</span> <span>${escape_html(store_get($$store_subs ??= {}, "$m", metricsState).cpuPercent.toFixed(2))}%</span></div> `;
            Progressbar($$payload3, {
              animate: true,
              progress: store_get($$store_subs ??= {}, "$m", metricsState).cpuPercent,
              color: "blue"
            });
            $$payload3.out += `<!----></li> <li><div class="flex items-center justify-between"><span class="font-medium">RAM Usage:</span> <span>${escape_html(store_get($$store_subs ??= {}, "$m", metricsState).ramPercent.toFixed(2))}%</span></div> `;
            Progressbar($$payload3, {
              animate: true,
              progress: store_get($$store_subs ??= {}, "$m", metricsState).ramPercent,
              color: "green"
            });
            $$payload3.out += `<!----></li></ul>`;
          },
          $$slots: { default: true }
        });
        $$payload2.out += `<!----></div>`;
      },
      $$slots: { default: true }
    });
    $$payload.out += `<!----> `;
    Card($$payload, {
      class: "p-6",
      children: ($$payload2) => {
        $$payload2.out += `<h3 class="mb-4 text-xl font-semibold">GPU Metrics</h3> <div class="space-y-6">`;
        Card($$payload2, {
          class: "p-4",
          children: ($$payload3) => {
            $$payload3.out += `<h4 class="font- mb-2 align-middle text-lg">Global</h4> <ul class="space-y-4"><li><span class="font-medium">Name:</span> ${escape_html(store_get($$store_subs ??= {}, "$m", metricsState).gpuName)}</li> <li><span class="font-medium">Temperature:</span> ${escape_html(store_get($$store_subs ??= {}, "$m", metricsState).gpuTemperature)}°C</li> <li><div class="flex items-center justify-between"><span class="font-medium">Utilization:</span> <span>${escape_html(store_get($$store_subs ??= {}, "$m", metricsState).gpuUtilizationPercent)}%</span></div> `;
            Progressbar($$payload3, {
              animate: true,
              progress: store_get($$store_subs ??= {}, "$m", metricsState).gpuUtilizationPercent,
              color: "purple"
            });
            $$payload3.out += `<!----></li> <li><div class="flex items-center justify-between"><span class="font-medium">Power Draw:</span> <span>${escape_html(store_get($$store_subs ??= {}, "$m", metricsState).gpuPowerDraw.toFixed(2))} W</span></div> `;
            Progressbar($$payload3, {
              animate: true,
              progress: store_get($$store_subs ??= {}, "$m", metricsState).gpuPowerDraw / store_get($$store_subs ??= {}, "$m", metricsState).gpuPowerLimit * 100,
              color: "purple"
            });
            $$payload3.out += `<!----></li> <li><div class="flex items-center justify-between"><span class="font-medium">VRAM Total:</span> <span>${escape_html((store_get($$store_subs ??= {}, "$m", metricsState).gpuVramTotal / 1024).toFixed(2))} GiB</span></div></li></ul>`;
          },
          $$slots: { default: true }
        });
        $$payload2.out += `<!----> `;
        Card($$payload2, {
          class: "p-4",
          children: ($$payload3) => {
            $$payload3.out += `<h4 class="mb-2 text-lg font-semibold">Simulation</h4> <ul class="space-y-4"><li><div class="flex items-center justify-between"><span class="font-medium">VRAM Used:</span> <span>${escape_html((store_get($$store_subs ??= {}, "$m", metricsState).gpuVramUsed / 1024).toFixed(2))} GiB</span></div> `;
            Progressbar($$payload3, {
              animate: true,
              progress: store_get($$store_subs ??= {}, "$m", metricsState).gpuVramUsed / store_get($$store_subs ??= {}, "$m", metricsState).gpuVramTotal * 100,
              color: "purple"
            });
            $$payload3.out += `<!----></li></ul>`;
          },
          $$slots: { default: true }
        });
        $$payload2.out += `<!----></div>`;
      },
      $$slots: { default: true }
    });
    $$payload.out += `<!----></div>`;
  }
  $$payload.out += `<!--]--></section>`;
  if ($$store_subs) unsubscribe_stores($$store_subs);
  pop();
}
function _page($$payload) {
  Header($$payload);
  $$payload.out += `<!----> <div class="grid-container svelte-1otn1km">`;
  Preview($$payload);
  $$payload.out += `<!----> `;
  TablePlot($$payload);
  $$payload.out += `<!----> `;
  Solver($$payload);
  $$payload.out += `<!----> `;
  Console($$payload);
  $$payload.out += `<!----> `;
  Mesh($$payload);
  $$payload.out += `<!----> `;
  Metrics($$payload);
  $$payload.out += `<!----> `;
  Parameters($$payload);
  $$payload.out += `<!----></div>`;
}
export {
  _page as default
};
