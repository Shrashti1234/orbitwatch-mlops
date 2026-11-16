const byId = (id) => document.getElementById(id);
const channelSelect = byId('channel');
const detectorSelect = byId('detector');
const overview = byId('overview');
const focus = byId('focus');
const number = new Intl.NumberFormat('en-US');
let activeReplay;
let selectedRange;
let pending;

function showError(message) {
  byId('status').textContent = message;
  byId('status').hidden = !message;
}

async function getJson(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`Request failed (${response.status}).`);
  return response.json();
}

function draw(canvas, values, title, threshold, ticks) {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const [first, last] = selectedRange || [0, values.length - 1];
  const samples = values.slice(first, last + 1).filter(Number.isFinite);
  if (threshold !== undefined) samples.push(threshold);
  let min = samples.length ? Math.min(...samples) : 0;
  let max = samples.length ? Math.max(...samples) : 1;
  const margin = (max - min) * 0.09 || 0.1;
  min -= margin;
  max += margin;
  const left = 54;
  const right = Math.max(left + 1, width - 18);
  const top = 26;
  const bottom = height - (ticks ? 23 : 6);
  const x = (index) => left + (index - first) * (right - left) / Math.max(last - first, 1);
  const y = (value) => bottom - (value - min) * (bottom - top) / (max - min);

  ctx.font = '12px Segoe UI, sans-serif';
  ctx.fillStyle = '#d5e1f7';
  ctx.fillText(title, left, 16);
  ctx.fillStyle = '#0d1526';
  ctx.fillRect(left, top, right - left, bottom - top);
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let i = 0; i < 4; i += 1) {
    const value = min + (max - min) * i / 3;
    const point = y(value);
    ctx.fillStyle = '#9fadc5';
    ctx.fillText(Number(value.toPrecision(2)).toString(), left - 8, point);
    ctx.fillStyle = '#273750';
    ctx.fillRect(left, point, right - left, 1);
  }
  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, right - left, bottom - top);
  ctx.clip();
  for (const [intervals, color] of [
    [activeReplay.labeled_intervals, 'rgba(242, 190, 96, .20)'],
    [activeReplay.alerts, 'rgba(255, 130, 147, .55)'],
  ]) {
    ctx.fillStyle = color;
    for (const [start, end] of intervals) ctx.fillRect(x(start), top, Math.max(2, x(end + 1) - x(start)), bottom - top);
  }
  if (threshold !== undefined) {
    ctx.beginPath();
    ctx.strokeStyle = '#f2be60';
    ctx.setLineDash([5, 4]);
    ctx.moveTo(left, y(threshold));
    ctx.lineTo(right, y(threshold));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.beginPath();
  ctx.strokeStyle = '#70e1d0';
  ctx.lineWidth = 1.15;
  let visible = false;
  for (let index = first; index <= last; index += 1) {
    const value = values[index];
    if (!Number.isFinite(value)) { visible = false; continue; }
    if (visible) ctx.lineTo(x(index), y(value));
    else ctx.moveTo(x(index), y(value));
    visible = true;
  }
  ctx.stroke();
  ctx.restore();
  if (ticks) {
    ctx.fillStyle = '#9fadc5';
    ctx.textAlign = 'center';
    for (let i = 0; i < 5; i += 1) {
      const index = Math.round(first + (last - first) * i / 4);
      ctx.fillText(number.format(index), x(index), bottom + 13);
    }
  }
}

function redraw() {
  if (!activeReplay) return;
  draw(byId('readings'), activeReplay.readings, 'Normalized sensor reading');
  draw(byId('scores'), activeReplay.scores, 'Anomaly score (higher means more unusual)', activeReplay.threshold, true);
}

function setRange(interval) {
  selectedRange = interval && [Math.max(0, interval[0] - 115), Math.min(activeReplay.readings.length - 1, interval[1] + 115)];
  overview.setAttribute('aria-pressed', String(!selectedRange));
  focus.setAttribute('aria-pressed', String(Boolean(selectedRange)));
  const range = selectedRange || [0, activeReplay.readings.length - 1];
  byId('range-label').textContent = `${activeReplay.channel} · Samples ${number.format(range[0])}–${number.format(range[1])}`;
  redraw();
}

function renderAlerts() {
  const rows = activeReplay.alerts.map(([start, end]) => {
    const row = document.createElement('tr');
    const cell = row.insertCell();
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = number.format(start);
    button.setAttribute('aria-label', `Inspect alert starting at sample ${number.format(start)}`);
    button.addEventListener('click', () => setRange([start, end]));
    cell.append(button);
    row.insertCell().textContent = number.format(end);
    row.insertCell().textContent = `${number.format(end - start + 1)} samples`;
    const overlap = activeReplay.labeled_intervals.some(([labelStart, labelEnd]) => start <= labelEnd && end >= labelStart);
    row.insertCell().textContent = overlap ? 'Yes' : 'No';
    return row;
  });
  byId('alerts').replaceChildren(...rows);
  byId('no-alerts').hidden = rows.length > 0;
}

async function loadChannel(channel, detector) {
  if (pending) pending.abort();
  pending = new AbortController();
  try {
    const record = await getJson(`/api/replay/${encodeURIComponent(channel)}?detector=${encodeURIComponent(detector)}`, pending.signal);
    const metrics = record.metrics;
    activeReplay = record;
    byId('samples').textContent = number.format(record.readings.length);
    byId('detected').textContent = `${metrics.events_detected} of ${metrics.events_labeled}`;
    byId('unmatched').textContent = number.format(metrics.unmatched_alert_intervals);
    byId('outside').textContent = number.format(metrics.alerted_samples_outside_labels);
    renderAlerts();
    setRange(null);
    overview.disabled = false;
    focus.disabled = !record.labeled_intervals.length;
    showError('');
  } catch (error) {
    if (error.name !== 'AbortError') {
      if (activeReplay) {
        channelSelect.value = activeReplay.channel;
        detectorSelect.value = activeReplay.detector;
      }
      showError(`Unable to load channel ${channel}. ${error.message}`);
    }
  }
}

async function start() {
  try {
    const { channels } = await getJson('/api/channels');
    channelSelect.replaceChildren(...channels.map((id) => new Option(id, id)));
    channelSelect.value = 'A-3';
    channelSelect.disabled = false;
    detectorSelect.disabled = false;
    await loadChannel(channelSelect.value, detectorSelect.value);
  } catch (error) {
    showError(`Unable to load the dashboard. ${error.message}`);
  }
}

channelSelect.addEventListener('change', () => loadChannel(channelSelect.value, detectorSelect.value));
detectorSelect.addEventListener('change', () => loadChannel(channelSelect.value, detectorSelect.value));
overview.addEventListener('click', () => setRange(null));
focus.addEventListener('click', () => setRange(activeReplay.labeled_intervals[0]));
new ResizeObserver(redraw).observe(document.querySelector('.chart'));
start();
