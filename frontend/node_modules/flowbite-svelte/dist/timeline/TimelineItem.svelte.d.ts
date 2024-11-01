import { SvelteComponentTyped } from "svelte";
declare const __propDef: {
    props: {
        title?: string;
        date?: string;
        svgClass?: string;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        icon: {};
        default: {};
    };
};
export type TimelineItemProps = typeof __propDef.props;
export type TimelineItemEvents = typeof __propDef.events;
export type TimelineItemSlots = typeof __propDef.slots;
/**
 * [Go to docs](https://flowbite-svelte.com/)
 * ## Props
 * @prop export let title: $$Props['title'] = '';
 * @prop export let date: $$Props['date'] = '';
 * @prop export let svgClass: $$Props['svgClass'] = 'w-3 h-3 text-primary-600 dark:text-primary-400';
 */
export default class TimelineItem extends SvelteComponentTyped<TimelineItemProps, TimelineItemEvents, TimelineItemSlots> {
}
export {};
