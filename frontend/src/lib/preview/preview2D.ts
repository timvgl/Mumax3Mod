import { previewState, type Preview } from '$api/incoming/preview';
import { getNewQuantityState, setNewQuantityFalse } from '$api/outgoing/preview';
import * as echarts from 'echarts';
import { get } from 'svelte/store';
import { meshState } from '$api/incoming/mesh';
import { disposePreview3D } from './preview3D';

export function preview2D() {
	if (get(previewState).scalarField === null) {
		disposePreview2D();
		disposePreview3D();
		return;
	}

	const container = document.getElementById('container')!;
	if (get(previewState).refresh || !container) {
		disposePreview2D();
		disposePreview3D();
		init();
	} else {
		update();
	}
}

function getColorMap(min: number, max: number) {
	if (min < 0 && max > 0) {
		return ['#313695', '#ffffff', '#a50026'];
	} else {
		return ['#ffffff', '#a50026'];
	}
}

let chartInstance: echarts.ECharts;

function update() {
	if (chartInstance === undefined || chartInstance.isDisposed()) {
		init();
		return;
	}
	let data = get(previewState).scalarField;
	let ps = get(previewState);
	chartInstance.setOption({
		tooltip: {
			formatter: function (params: any) {
				if (params.value === undefined) {
					return 'NaN';
				}
				return `${params.value[2]} ${ps.unit}`;
			}
		},
		dataset: {
			source: data
		},
		visualMap: [
			{
				max: ps.max,
				min: ps.min
			}
		]
	});
	if (getNewQuantityState()) {
		let isFFT = (get(previewState).dynQuantities["FFT"] ?? []).includes(get(previewState).quantity)
		let [xData, yData, xAxisName, yAxisName] = get_axis_data(ps, isFFT)
		let option = chartInstance.getOption();
		if (Array.isArray(option.xAxis)) {
			option.xAxis[0].data = xData;
			option.xAxis[0].name = xAxisName;
			if (option.xAxis[0].axisTick && option.xAxis[0].axisTick.interval) {
				option.xAxis[0].axisTick.interval = ps.symmetricX ? symmetricTickIntervalTicks(xData) : function (index: number) {
					// Show ticks only at every 10th value
					return index % 10 === 0;
				}	
			}
			if (option.xAxis[0].axisLabel) {
				option.xAxis[0].axisLabel.formatter = ps.symmetricX ? function (value: string, index: number) {
					return symmetricTickIntervalLabel(xData)(index) ? parseFloat(value).toFixed(0) : '';
				} : function (value: string, index: number) {
					return index % 10 === 0 ? parseFloat(value).toFixed(0) : '';
				}
				option.xAxis[0].axisLabel.interval = ps.symmetricX ? symmetricTickIntervalTicks(xData) : function (index: number) {
					// Show ticks only at every 10th value
					return index % 10 === 0;
				}
			}
		} else if (option.xAxis) {
			option.xAxis.data = xData;
			option.xAxis.name = xAxisName;
			if (option.xAxis.axisTick && option.xAxis.axisTick.interval) {
				option.xAxis.axisTick.interval = ps.symmetricX ? symmetricTickIntervalTicks(xData) : function (index: number) {
					// Show ticks only at every 10th value
					return index % 10 === 0;
				}	
			}
			if (option.xAxis.axisLabel) {
				option.xAxis.axisLabel.formatter = ps.symmetricX ? function (value: string, index: number) {
					return symmetricTickIntervalLabel(xData)(index) ? parseFloat(value).toFixed(0) : '';
				} : function (value: string, index: number) {
					return index % 10 === 0 ? parseFloat(value).toFixed(0) : '';
				}
				option.xAxis.axisLabel.interval = ps.symmetricX ? symmetricTickIntervalTicks(xData) : function (index: number) {
					// Show ticks only at every 10th value
					return index % 10 === 0;
				}
			}
		}
		
		if (Array.isArray(option.yAxis)) {
			option.yAxis[0].data = yData;
			option.yAxis[0].name = yAxisName;
			if (option.yAxis[0].axisTick && option.yAxis[0].axisTick.interval) {
				option.yAxis[0].axisTick.interval = ps.symmetricY ? symmetricTickIntervalTicks(yData) : function (index: number) {
					// Show ticks only at every 10th value
					return index % 10 === 0;
				}	
			}
			if (option.yAxis[0].axisLabel) {
				option.yAxis[0].axisLabel.formatter = ps.symmetricY ? function (value: string, index: number): string {
					return symmetricTickIntervalLabel(yData)(index) ? parseFloat(value).toFixed(0) : '';
				} : function (value: string, index: number) {
					return index % 10 === 0 ? parseFloat(value).toFixed(0) : '';
				}
				option.yAxis[0].axisLabel.interval = ps.symmetricY ? symmetricTickIntervalTicks(yData) : function (index: number) {
					// Show ticks only at every 10th value
					return index % 10 === 0;
				}
			}
		} else if (option.yAxis) {
			option.yAxis.data = yData;
			option.yAxis.name = yAxisName;
			if (option.yAxis.axisTick && option.yAxis.axisTick.interval) {
				option.yAxis.axisTick.interval = ps.symmetricY ? symmetricTickIntervalTicks(yData) : function (index: number) {
					// Show ticks only at every 10th value
					return index % 10 === 0;
				}	
			}
			if (option.yAxis.axisLabel) {
				option.yAxis.axisLabel.interval = ps.symmetricY ? symmetricTickIntervalTicks(yData) : function (index: number) {
					// Show ticks only at every 10th value
					return index % 10 === 0;
				}
				option.yAxis.axisLabel.formatter = ps.symmetricY ? function (value: string, index: number): String {
					return symmetricTickIntervalLabel(yData)(index) ? parseFloat(value).toFixed(0) : '';
				} : function (value: string, index: number) {
					return index % 10 === 0 ? parseFloat(value).toFixed(0) : '';
				}
			}
		}

		if (option.axisPointer && option.axisPointer.label) {
			option.axisPointer.label.formatter =  function (params: any) {
				if (params.value === undefined) {
					return 'NaN';
				}
				let unit = isFFT ? "1/μm" : "nm"
				return ` ${parseFloat(params.value).toFixed(0)} ${unit}`;
			}
		}

        // Set the updated option back
        chartInstance.setOption(option);
		setNewQuantityFalse();
	}
}

function symmetricTickIntervalLabel(data: any[]): (index: number) => boolean {
    const total = data.length;

    if (total < 5) {
        return () => true; // Show all ticks if data is too small
    }

    const low = 0;
    const high = total - 1;
    const mid = Math.floor(total / 2);
    const quarter = Math.floor((mid - low) / 2);
    const threeQuarter = mid + quarter;

    // Return a function to align with ECharts' scaling
    return (index: number) => [low, quarter, mid, threeQuarter, high].includes(index);
}

function symmetricTickIntervalTicks(data: any[]): (index: number) => boolean {
    const total = data.length;

    // If there's too little data, just show all ticks
    if (total < 5) {
        return () => true;
    }

    // Key indices
    const low = 0;
    const high = total - 1;
    const mid = Math.floor(total / 2);
    const quarter = Math.floor((mid - low) / 2);
    const threeQuarter = mid + quarter;

    // This function subdivides an interval [start, end] into (intermediates + 1) segments,
    // returning all intermediate points including start and end.
    // For example, if intermediates = 3 and start=0, end=10:
    // It returns [0, 2, 4, 6, 8, 10] (5 segments = 4 intervals).
    function subdivideInterval(start: number, end: number, intermediates: number): number[] {
        const step = (end - start) / (intermediates + 1);
        const points: number[] = [];
        for (let i = 0; i <= intermediates + 1; i++) {
            // Use Math.round to ensure indices remain integers
            points.push(Math.round(start + step * i));
        }
        return points;
    }

    // Main intervals to subdivide
    const intervals = [
        [low, quarter],
        [quarter, mid],
        [mid, threeQuarter],
        [threeQuarter, high]
    ];

    let ticks: number[] = [];

    // Subdivide each interval and combine them
    intervals.forEach((interval, index) => {
        let [start, end] = interval;
        let subdivided = subdivideInterval(start, end, 3);
        // Avoid duplicating endpoints: after the first interval, remove the first point
        // because it was the end of the previous interval.
        if (index > 0) {
            subdivided.shift();
        }
        ticks = ticks.concat(subdivided);
    });

    // Convert to a set for O(1) lookups and ensure uniqueness
    const tickSet = new Set(ticks);

    // Return a function that checks if the given index is a tick
    return (index: number) => tickSet.has(index);
}


function get_axis_data(ps: Preview, isFFT: boolean): [string[], string[], string, string] {
	let dims = [ps.xChosenSize, ps.yChosenSize];
	let mesh = get(meshState);
	let xStart = ps.startX;
	let yStart = ps.startY;
	let xData;
	let yData;
	let xAxisName = "";
	let yAxisName = "";
	// console.log(mesh.Nx/ps.xChosenSize);
	if (isFFT === true) {
		xData = Array.from({ length: dims[0] }, (_, i) => String((xStart * 1e-6 + i * mesh.dx * 1e-6) * (mesh.Nx) / ps.xChosenSize));
		yData = Array.from({ length: dims[1] }, (_, i) => String((yStart * 1e-6 + i * mesh.dy * 1e-6) * (mesh.Ny) / ps.yChosenSize));
		xAxisName = "kx (1/nm)"
		yAxisName = "ky (1/nm)"
	} else {
		xData = Array.from({ length: dims[0] }, (_, i) => String((xStart * 1e9 + i * mesh.dx * 1e9) * mesh.Nx / ps.xChosenSize));
		yData = Array.from({ length: dims[1] }, (_, i) => String((yStart * 1e9 + i * mesh.dy * 1e9) * mesh.Ny / ps.yChosenSize));
		xAxisName = "x (nm)"
		yAxisName = "y (nm)"
	}
	return [xData, yData, xAxisName, yAxisName]
}

function init() {
	var chartDom = document.getElementById('container')!;
	// https://apache.github.io/echarts-handbook/en/best-practices/canvas-vs-svg
	chartInstance = echarts.init(chartDom, undefined, { renderer: 'svg' });
	let ps = get(previewState);
	let isFFT = (get(previewState).dynQuantities["FFT"] ?? []).includes(get(previewState).quantity);
	let [xData, yData, xAxisName, yAxisName] = get_axis_data(ps, isFFT);

	let aspectRatio = ps.xChosenSize / ps.yChosenSize; // Calculate the aspect ratio

	let gridWidth, gridHeight;

	// Ensure the grid dimensions maintain the aspect ratio
	if (aspectRatio > 1) {
		gridWidth = '80%';
		gridHeight = `${80 / aspectRatio}%`;
	} else {
		gridWidth = `${80 * aspectRatio}%`;
		gridHeight = '80%';
	}

	let data = get(previewState).scalarField;

	// @ts-ignore
	chartInstance.setOption({
		tooltip: {
			position: 'top',
			formatter: function (params: any) {
				if (params.value === undefined) {
					return 'NaN';
				}
				return `${params.value[2]} ${ps.unit}`;
			},
			backgroundColor: '#282a36',
			borderColor: '#6e9bcb',
			textStyle: {
				color: '#fff'
			}
		},
		axisPointer: {
			show: true,
			type: 'line',
			triggerEmphasis: false,
			lineStyle: {
				color: '#6e9bcb',
				width: 2,
				type: 'dashed'
			},
			label: {
				backgroundColor: '#282a36',
				color: '#fff',
				formatter: function (params: any) {
					if (params.value === undefined) {
						return 'NaN';
					}
					let unit = isFFT ? "1/μm" : "nm"
					return ` ${parseFloat(params.value).toFixed(0)} ${unit}`;
				},
				padding: [8, 5, 8, 5],
				borderColor: '#6e9bcb',
				borderWidth: 1
			}
		},
		xAxis: {
			type: 'category',
			data: xData,
			name: xAxisName,
			nameLocation: 'middle',
			nameGap: 25,
			nameTextStyle: {
				color: '#fff'
			},
			axisTick: {
				alignWithLabel: true,
				interval: ps.symmetricX ? symmetricTickIntervalTicks(xData) : function (index: number) {
					// Show ticks only at every 10th value
					return index % 10 === 0;
				},
				length: 6,
				lineStyle: {
					type: 'solid',
					color: '#fff'
				}
			},
			axisLabel: {
				show: true,
				interval: ps.symmetricX ? symmetricTickIntervalTicks(xData) : function (index: number) {
					// Show ticks only at every 10th value
					return index % 10 === 0;
				},
				formatter: ps.symmetricX ? function (value: string, index: number) {
					return symmetricTickIntervalLabel(xData)(index) ? parseFloat(value).toFixed(0) : '';
				} : function (value: string, index: number) {
					return index % 10 === 0 ? parseFloat(value).toFixed(0) : '';
				},
				color: '#fff',
				showMinLabel: true
			}
		},
		yAxis: {
			type: 'category',
			data: yData,
			name: yAxisName,
			nameLocation: 'middle',
			nameGap: 45,
			nameTextStyle: {
				color: '#fff'
			},
			axisTick: {
				alignWithLabel: true,
				interval: ps.symmetricY ? symmetricTickIntervalTicks(yData) : function (index: number) {
					// Show ticks only at every 10th value
					return index % 10 === 0;
				},
				length: 6,
				lineStyle: {
					type: 'solid',
					color: '#fff'
				}
			},
			axisLabel: {
				show: true,
				interval: ps.symmetricY ? symmetricTickIntervalTicks(yData) : function (index: number) {
					// Show ticks only at every 10th value
					return index % 10 === 0;
				},
				formatter: ps.symmetricY ? function (value: string, index: number): String{
					return symmetricTickIntervalLabel(yData)(index) ? parseFloat(value).toFixed(0) : '';
				} : function (value: string, index: number) {
					return index % 10 === 0 ? parseFloat(value).toFixed(0) : '';
				},
				color: '#fff',
				showMinLabel: true
			}
		},
		visualMap: [
			{
				type: 'continuous',
				min: ps.min,
				max: ps.max,
				calculable: true,
				realtime: true,
				// precision: 2,
				formatter: function (value: any) {
					return parseFloat(value).toPrecision(2);
				},
				itemWidth: 9,
				itemHeight: 140,
				text: [ps.unit, ''],
				textStyle: {
					color: '#fff'
				},
				inRange: {
					color: getColorMap(ps.min, ps.max)
				},
				left: 'right'
			}
		],
		series: [
			{
				name: ps.quantity,
				type: 'heatmap',
				emphasis: {
					itemStyle: {
						borderColor: '#333',
						borderWidth: 1
					}
				},
				selectedMode: false,
				progressive: 0,
				animation: true
			}
		],
		dataset: {
			source: data
		},
		grid: {
			containLabel: false,
			left: '10%',
			right: '10%'
		},
		toolbox: {
			show: true,
			itemSize: 20,
			iconStyle: {
				borderColor: '#fff'
			},
			feature: {
				dataZoom: {
					xAxisIndex: 0,
					yAxisIndex: 0,
					brushStyle: {
						color: '#282a3655',
						borderColor: '#6e9bcb',
						borderWidth: 2
					}
				},
				dataView: { show: false },
				restore: {
					show: false
				},
				saveAsImage: {
					type: 'png',
					name: 'preview'
				}
			}
		}
	});
}

export function disposePreview2D() {
	const container = document.getElementById('container')!;
	if (container) {
		const echartsInstance = echarts.getInstanceByDom(container);
		if (echartsInstance) {
			echartsInstance.dispose();
		}
	}
}

export function resizeECharts() {
	window.addEventListener('resize', function () {
		if (chartInstance === undefined || chartInstance.isDisposed()) {
			return;
		}
	});
}