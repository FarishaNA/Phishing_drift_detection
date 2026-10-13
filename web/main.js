(function () {
  'use strict';
  const $ = (s, r) => (r || document).querySelector(s);
  const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const PD = window.PD || {};
  const LABEL = PD.modelLabel || { NB: 'Naive Bayes', SVM: 'SVM', RF: 'Random Forest' };
  const page = document.body.dataset.page;
  const ICON_SHIELD = '<svg viewBox="0 0 24 24"><path d="M12 3l8 3v6c0 4.5-3.2 7.8-8 9-4.8-1.2-8-4.5-8-9V6l8-3z"/><path d="M8.8 12l2.2 2.2 4.2-4.4"/></svg>';
  const ICON_MAIL = '<svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="2.5"/><path d="M3.5 7.5l8.5 6 8.5-6"/></svg>';

  /* ---------------- generic tabs (evidence, about, drift) ---------------- */
  $$('[data-tabset]').forEach((set) => {
    const btns = $$('.tab', set);
    const panes = $$('.pane[data-pane]', set.parentElement);
    btns.forEach((b) => b.addEventListener('click', () => {
      btns.forEach((x) => x.classList.toggle('active', x === b));
      panes.forEach((p) => p.classList.toggle('hidden', p.dataset.pane !== b.dataset.pane));
    }));
  });

  /* ---------------- history store ---------------- */
  const HKEY = 'phishdrift_history_v1';
  const loadHistory = () => { try { return JSON.parse(localStorage.getItem(HKEY) || '[]'); } catch (e) { return []; } };
  const saveHistory = (rows) => { try { localStorage.setItem(HKEY, JSON.stringify(rows.slice(0, 200))); } catch (e) { /* storage blocked */ } };

  /* ================= HOME ================= */
  function initHome() {
    const raw = $('#raw'), count = $('#count'), ex = $('#example'), truth = $('#truth');
    const subj = $('#subject'), sender = $('#sender'), fileIn = $('#file'), info = $('#file-info'), drop = $('#drop');
    const result = $('#result');
    let uploaded = null;

    raw.addEventListener('input', () => { count.textContent = raw.value.length + '/5000'; });

    ex.addEventListener('change', () => {
      const s = (PD.samples || {})[ex.value];
      if (!s) { truth.classList.add('hidden'); return; }
      subj.value = s.subject; sender.value = s.sender; raw.value = s.body;
      count.textContent = raw.value.length + '/5000';
      truth.textContent = 'Example ground truth: ' + s.truth;
      truth.classList.remove('hidden');
    });

    $$('.tab[data-tab]').forEach((t) => t.addEventListener('click', () => {
      $$('.tab[data-tab]').forEach((x) => x.classList.toggle('active', x === t));
      $('#pane-paste').classList.toggle('hidden', t.dataset.tab !== 'paste');
      $('#pane-upload').classList.toggle('hidden', t.dataset.tab !== 'upload');
    }));

    function setFile(f) {
      if (!f) return;
      if (f.size > 200000) { info.innerHTML = '<span class="v-bad">File is larger than 200 KB.</span>'; return; }
      f.text().then((txt) => {
        uploaded = { name: f.name, text: txt };
        info.innerHTML = '<b>' + esc(f.name) + '</b> loaded (' + Math.round(f.size / 1024 * 10) / 10 + ' KB). It will be used instead of pasted text. ' +
          '<a href="#" id="rm-file">Remove</a>';
        $('#rm-file').addEventListener('click', (e) => { e.preventDefault(); uploaded = null; fileIn.value = ''; info.textContent = ''; });
      });
    }
    fileIn.addEventListener('change', () => setFile(fileIn.files[0]));
    ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
    ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
    drop.addEventListener('drop', (e) => { if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });

    $('#clear').addEventListener('click', () => {
      raw.value = ''; subj.value = ''; sender.value = ''; ex.value = ''; count.textContent = '0/5000';
      truth.classList.add('hidden'); uploaded = null; fileIn.value = ''; info.textContent = '';
      result.innerHTML = emptyState();
    });

    $('#classify').addEventListener('click', async () => {
      const payload = uploaded
        ? { kind: uploaded.name.toLowerCase().endsWith('.eml') ? 'eml' : 'text', text: uploaded.text, subject: '', sender: '' }
        : { kind: 'text', text: raw.value, subject: subj.value, sender: sender.value };
      if (!payload.text.trim() && !payload.subject.trim()) {
        result.innerHTML = '<div class="callout warn">Paste an email or upload a file first.</div>' + emptyState();
        return;
      }
      const btn = $('#classify'); btn.disabled = true;
      result.innerHTML = '<div class="loading"><div class="spin"></div>Analyzing email...</div>';
      try {
        const resp = await fetch('/api/classify', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || 'Request failed');
        renderResult(data);
        addHistory(data);
      } catch (err) {
        result.innerHTML = '<div class="callout bad"><b>Could not classify.</b> ' + esc(err.message) + '</div>';
      } finally { btn.disabled = false; }
    });

    function emptyState() {
      return '<div class="empty"><div class="ic">' + ICON_MAIL + '</div><div class="e1">No email analyzed yet</div>' +
        '<div class="e2">Paste an email or upload a file to get classification results.</div></div>';
    }

    function addHistory(data) {
      const rows = loadHistory();
      const r = data.results, order = data.order;
      const nph = order.filter((m) => r[m].label === 'phishing').length;
      const verdict = (nph * 2 > order.length ? 'Phishing' : nph === 0 ? 'Legitimate' : 'Mixed') + ' (' + nph + '/' + order.length + ' flag phishing)';
      const row = { time: new Date().toLocaleString(), subject: (data.subject || '(no subject)').slice(0, 70), from: (data.sender || '(unknown)').slice(0, 50), verdict: verdict, models: {} };
      order.forEach((m) => { row.models[m] = r[m].label + ' ' + Math.round(r[m].p_phish * 100) + '%'; });
      rows.unshift(row); saveHistory(rows);
    }

    function termsHtml(t) {
      if (t.directional) {
        const mx = Math.max.apply(null, t.phish.concat(t.legit).map((x) => Math.abs(x[1])).concat([1e-9]));
        const rows = (list, cls) => list.length ? list.map((x) => '<div class="row ' + cls + '"><span title="' + esc(x[0]) + '">' + esc(x[0]) +
          '</span><div class="bar"><i style="width:' + Math.max(4, Math.abs(x[1]) / mx * 100) + '%"></i></div><span>' + Math.abs(x[1]).toFixed(2) + '</span></div>').join('') :
          '<div class="hint">None</div>';
        return '<div class="tb"><h5>Pushes toward phishing</h5>' + rows(t.phish, 'ph') + '<h5>Pushes toward legit</h5>' + rows(t.legit, 'lg') + '</div>';
      }
      const items = t.influential || [];
      const mx = Math.max.apply(null, items.map((x) => x[1]).concat([1e-9]));
      return '<div class="tb"><h5>Most influential terms (Random Forest gives no direction)</h5>' + (items.length ? items.map((x) => '<div class="row ng"><span title="' +
        esc(x[0]) + '">' + esc(x[0]) + '</span><div class="bar"><i style="width:' + Math.max(4, x[1] / mx * 100) + '%"></i></div><span>' + x[1].toFixed(3) + '</span></div>').join('') : '<div class="hint">None</div>') + '</div>';
    }

    function renderResult(data) {
      const r = data.results, order = data.order, n = order.length;
      const nph = order.filter((m) => r[m].label === 'phishing').length;
      const bad = nph * 2 > n, tied = nph * 2 === n;
      const tone = bad ? 'bad' : tied ? '' : 'good';
      const title = bad ? 'Likely phishing' : tied ? 'Models disagree' : 'Likely legitimate';
      const sub = (bad || tied) ? nph + ' of ' + n + ' classifiers flag this email as phishing' : (n - nph) + ' of ' + n + ' classifiers consider this email legitimate';
      let h = '<div class="result ' + tone + '"><div class="k">Verdict</div><div class="t">' + title + '</div><div class="s">' + sub + '</div></div>';
      h += '<table class="rt"><tr><th>Model</th><th>Verdict</th><th>Phishing prob.</th></tr>';
      order.forEach((m) => {
        const x = r[m], ph = x.label === 'phishing';
        h += '<tr><td>' + esc(LABEL[m]) + '</td><td class="' + (ph ? 'v-bad' : 'v-good') + '">' + (ph ? 'Phishing' : 'Legitimate') + '</td><td><span class="mini"><i style="width:' +
          Math.round(x.p_phish * 100) + '%"></i></span>' + Math.round(x.p_phish * 100) + '%</td></tr>';
      });
      h += '</table>';
      const unf = order.filter((m) => r[m].unfamiliar).map((m) => LABEL[m]);
      h += unf.length ? '<div class="callout warn"><b>Unfamiliar input</b> for ' + esc(unf.join(', ')) + '. The email sits far from the training data (Df above 1), so treat the verdict with caution.</div>'
        : '<div class="callout good"><b>Drift check:</b> all models see this email as familiar (Df at or below 1).</div>';
      h += '<details class="more"><summary>Why these verdicts?</summary><div class="more-body"><div class="mtabs">' +
        order.map((m, i) => '<button class="mtab' + (i === 0 ? ' on' : '') + '" data-m="' + m + '">' + esc(LABEL[m]) + '</button>').join('') + '</div>' +
        order.map((m, i) => '<div class="mpane' + (i === 0 ? '' : ' hidden') + '" data-m="' + m + '">' + termsHtml(r[m].terms) + '<div class="hint" style="margin-top:8px">Drift score Df = ' + r[m].Df.toFixed(2) + '</div></div>').join('') +
        '</div></details>';
      const f = r[order[0]].features;
      h += '<details class="more"><summary>Features extracted</summary><div class="more-body"><table class="kv">' +
        Object.keys(f).map((k) => '<tr><td>' + esc(k) + '</td><td>' + esc(Math.round(f[k] * 1000) / 1000) + '</td></tr>').join('') + '</table></div></details>';
      result.innerHTML = h;
      $$('.mtab', result).forEach((b) => b.addEventListener('click', () => {
        $$('.mtab', result).forEach((x) => x.classList.toggle('on', x === b));
        $$('.mpane', result).forEach((p) => p.classList.toggle('hidden', p.dataset.m !== b.dataset.m));
      }));
    }
  }

  /* ================= HISTORY ================= */
  function initHistory() {
    const box = $('#hist');
    function draw() {
      const rows = loadHistory();
      if (!rows.length) {
        box.innerHTML = '<div class="card"><div class="empty"><div class="ic">' + ICON_MAIL + '</div><div class="e1">No history yet</div>' +
          '<div class="e2">Emails you classify on the Home page will be listed here.</div></div></div>';
        $('#clear-hist').classList.add('hidden');
        return;
      }
      $('#clear-hist').classList.remove('hidden');
      const cols = Object.keys(rows[0].models || {});
      let h = '<table class="dt"><tr><th>Time</th><th>Subject</th><th>From</th><th>Verdict</th>' + cols.map((m) => '<th>' + esc(LABEL[m] || m) + '</th>').join('') + '</tr>';
      rows.forEach((r) => {
        h += '<tr><td>' + esc(r.time) + '</td><td>' + esc(r.subject) + '</td><td>' + esc(r.from) + '</td><td>' + esc(r.verdict) + '</td>' +
          cols.map((m) => '<td>' + esc((r.models || {})[m] || '') + '</td>').join('') + '</tr>';
      });
      box.innerHTML = h + '</table>';
    }
    $('#clear-hist').addEventListener('click', () => { saveHistory([]); draw(); });
    draw();
  }

  /* ================= DRIFT ================= */
  function initDrift() {
    const years = PD.years, state = { model: PD.models[0], idx: 0, retrained: false };
    const slider = $('#yr-slider'), yrLabel = $('#yr-label');
    slider.max = years.length - 1;

    function seg(id, onPick) {
      $$('#' + id + ' button').forEach((b) => b.addEventListener('click', () => {
        $$('#' + id + ' button').forEach((x) => x.classList.toggle('on', x === b));
        onPick(b.dataset.v);
      }));
    }
    seg('seg-model', (v) => { state.model = v; refresh(); });
    seg('seg-state', (v) => { state.retrained = v === 'retrained'; refresh(); });
    $('#prev').addEventListener('click', () => { state.idx = Math.max(0, state.idx - 1); refresh(); });
    $('#next').addEventListener('click', () => { state.idx = Math.min(years.length - 1, state.idx + 1); refresh(); });
    $('#reset').addEventListener('click', () => { state.idx = 0; refresh(); });
    slider.addEventListener('input', () => { state.idx = +slider.value; refresh(); });

    const tone = { normal: 'good', 'abnormal drift': 'warn', 'retrain recommended': 'bad' };
    const pct = (x, d) => (x * 100).toFixed(d == null ? 1 : d) + '%';
    const stat = (l, v, s, t) => '<div class="stat"><div class="l">' + l + '</div><div class="v ' + (t || '') + '">' + v + '</div><div class="s">' + s + '</div></div>';

    async function refresh() {
      const year = years[state.idx];
      slider.value = state.idx; yrLabel.textContent = year;
      const q = 'model=' + state.model + '&year=' + year + '&retrained=' + (state.retrained ? 1 : 0);
      const r = await fetch('/api/drift?' + q); const d = await r.json();
      const t = tone[d.status] || '';
      $('#status').className = 'statusline ' + t;
      $('#status').innerHTML = '<span class="dot"></span><div><div class="m">' + year + ': ' + esc(d.status.charAt(0).toUpperCase() + d.status.slice(1)) + '</div><div class="d">' +
        pct(d.abn_pct) + ' of samples abnormal against a retrain threshold of ' + pct(d.thr) + '. Status is always measured against the original model.</div></div>';
      $('#stats').innerHTML = stat('Abnormal samples', pct(d.abn_pct), 'threshold ' + pct(d.thr), t) +
        stat('Phishing recall', d.recall.toFixed(3), state.retrained ? 'triggered retraining' : 'frozen model') +
        stat('Gap vs train-era', (d.gap_pts >= 0 ? '+' : '') + d.gap_pts.toFixed(1) + ' pts', 'control recall ' + d.control.toFixed(3), d.gap_pts > d.drop_margin_pts ? 'bad' : 'good') +
        stat('Mean drift score', d.mean_Df.toFixed(2), 'per-sample Df, batch mean');
      const stamp = '&t=' + Date.now();
      $('#img-drift').src = '/chart/drift.png?' + q + stamp;
      $('#img-recall').src = '/chart/recall.png?' + q + stamp;
      $('#img-hist').src = '/chart/hist.png?' + q + stamp;
      $('#hist-cap').textContent = pct(d.pct_abnormal_emails) + ' of the ' + d.n_emails + ' emails in ' + year + ' are abnormal (Df > 1).';
      const ev = d.retrain_events;
      $('#retrain-note').className = 'callout ' + (ev.length ? 'info' : 'warn');
      $('#retrain-note').innerHTML = ev.length ? 'Triggered policy retrained after: <b>' + ev.join(', ') + '</b>. The next year is scored by the retrained model.'
        : 'The detector never crossed its threshold for this model, so no retrain fired. The triggered policy behaves like Frozen. See Evidence.';
      $('#policy-rows').innerHTML = d.policies.map((p) => '<tr><td>' + esc(p.policy) + '</td><td>' + p.mean_recall.toFixed(3) + '</td><td>' + p.min_recall.toFixed(3) + '</td><td>' + p.n_retrains + '</td></tr>').join('');
    }
    refresh();
  }

  if (page === 'home') initHome();
  if (page === 'history') initHistory();
  if (page === 'drift') initDrift();
})();
