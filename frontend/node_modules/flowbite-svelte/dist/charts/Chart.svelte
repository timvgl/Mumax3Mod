<script>import { onMount } from "svelte";
export let options;
let chart;
function initChart(node, options2) {
  async function asyncInitChart() {
    const ApexChartsModule = await import("apexcharts");
    const ApexCharts = ApexChartsModule.default;
    chart = new ApexCharts(node, options2);
    await chart.render();
  }
  asyncInitChart();
  return {
    update(options3) {
      chart?.updateOptions(options3);
    },
    destroy() {
      chart?.destroy();
    }
  };
}
onMount(() => {
  return () => {
    chart?.destroy();
  };
});
</script>

<div use:initChart={options} class={$$props.class}></div>

<!--
@component
[Go to docs](https://flowbite-svelte.com/)
## Props
@prop export let options: ApexOptions;
-->
