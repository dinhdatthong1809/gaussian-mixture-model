/* Mô phỏng Gaussian Mixture Model bằng thuật toán EM.
   Toàn bộ chạy trong trình duyệt, không cần server. */

(function () {
  "use strict";

  // ------------------------------------------------------------------
  // Hằng số
  // ------------------------------------------------------------------
  var PALETTE = ["#d64550", "#2d6cdf", "#2fa84f", "#e08a1e", "#8b5cf6"];
  var DOMAIN = 10;          // dữ liệu nằm trong [0, 10] ở cả 2 trục
  var REG = 1e-4;           // cộng vào phương sai để ma trận không suy biến
  var W = 720, H = 720;     // kích thước canvas (đơn vị CSS pixel)
  var PAD = 48;

  var canvas = document.getElementById("canvas");
  var ctx = canvas.getContext("2d");

  var state = {
    dim: "2d",
    view: "ellipse",
    k: 3,
    pts1d: [],              // [{x, jitter}]
    pts2d: [],              // [{x, y}]
    model: null,            // 1D: {w, mu, va} | 2D: {w, mu, cov}
    resp: null,             // ma trận xác suất thuộc cụm [n][k]
    iter: 0,
    ll: null,
    prevLl: null,
    converged: false,
    autoTimer: null,
    message: ""
  };

  // ------------------------------------------------------------------
  // Tiện ích toán học
  // ------------------------------------------------------------------
  function randn() {
    var u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  function logSumExp(arr) {
    var max = -Infinity, i;
    for (i = 0; i < arr.length; i++) if (arr[i] > max) max = arr[i];
    if (max === -Infinity) return -Infinity;
    var sum = 0;
    for (i = 0; i < arr.length; i++) sum += Math.exp(arr[i] - max);
    return max + Math.log(sum);
  }

  function logNormal1d(x, mu, va) {
    var d = x - mu;
    return -0.5 * (Math.log(2 * Math.PI * va) + d * d / va);
  }

  function normal1d(x, mu, va) { return Math.exp(logNormal1d(x, mu, va)); }

  // Ma trận hiệp phương sai 2x2 lưu dạng {xx, xy, yy}
  function logNormal2d(x, y, mu, cov) {
    var det = cov.xx * cov.yy - cov.xy * cov.xy;
    if (det <= 1e-12) det = 1e-12;
    var dx = x - mu[0], dy = y - mu[1];
    // nghịch đảo ma trận 2x2
    var q = (cov.yy * dx * dx - 2 * cov.xy * dx * dy + cov.xx * dy * dy) / det;
    return -Math.log(2 * Math.PI) - 0.5 * Math.log(det) - 0.5 * q;
  }

  function normal2d(x, y, mu, cov) { return Math.exp(logNormal2d(x, y, mu, cov)); }

  /* Trị riêng / vector riêng của ma trận đối xứng 2x2 -> trục của ellipse */
  function eigen2d(cov) {
    var tr = cov.xx + cov.yy;
    var det = cov.xx * cov.yy - cov.xy * cov.xy;
    var disc = Math.sqrt(Math.max(tr * tr / 4 - det, 0));
    var l1 = tr / 2 + disc, l2 = tr / 2 - disc;
    var angle;
    if (Math.abs(cov.xy) > 1e-12) {
      angle = Math.atan2(l1 - cov.xx, cov.xy);
    } else {
      angle = cov.xx >= cov.yy ? 0 : Math.PI / 2;
    }
    return { l1: Math.max(l1, 1e-9), l2: Math.max(l2, 1e-9), angle: angle };
  }

  /* Pha màu về phía trắng -> dùng làm mặt sườn núi đục, che được hàng phía sau */
  function lighten(color, amount) {
    var m = /^rgb\((\d+),(\d+),(\d+)\)$/.exec(color);
    var r, g, b;
    if (m) { r = +m[1]; g = +m[2]; b = +m[3]; }
    else {
      r = parseInt(color.substr(1, 2), 16);
      g = parseInt(color.substr(3, 2), 16);
      b = parseInt(color.substr(5, 2), 16);
    }
    return "rgb(" + Math.round(r + (255 - r) * amount) + "," +
                    Math.round(g + (255 - g) * amount) + "," +
                    Math.round(b + (255 - b) * amount) + ")";
  }

  function mixColors(colors, weights) {
    var r = 0, g = 0, b = 0, i;
    for (i = 0; i < colors.length; i++) {
      var c = colors[i];
      r += weights[i] * parseInt(c.substr(1, 2), 16);
      g += weights[i] * parseInt(c.substr(3, 2), 16);
      b += weights[i] * parseInt(c.substr(5, 2), 16);
    }
    return "rgb(" + Math.round(r) + "," + Math.round(g) + "," + Math.round(b) + ")";
  }

  // ------------------------------------------------------------------
  // Dữ liệu
  // ------------------------------------------------------------------
  function points() { return state.dim === "1d" ? state.pts1d : state.pts2d; }

  function addPoints(dataX, dataY, count, spread) {
    var i;
    for (i = 0; i < count; i++) {
      if (state.dim === "1d") {
        state.pts1d.push({
          x: clamp(dataX + randn() * spread, 0, DOMAIN),
          jitter: Math.random()
        });
      } else {
        state.pts2d.push({
          x: clamp(dataX + randn() * spread, 0, DOMAIN),
          y: clamp(dataY + randn() * spread, 0, DOMAIN)
        });
      }
    }
    resetModel("Đã thêm điểm mới — hãy khởi tạo lại cụm.");
  }

  /* Điểm sôi động của 20 bài hát trong data/BaiHat.csv */
  var BAI_HAT = [2.9, 3.4, 3.8, 4.1, 4.4, 4.7, 5.1, 5.5, 5.6, 6.0,
                 6.3, 6.6, 6.8, 7.0, 7.2, 7.4, 7.7, 8.0, 8.3, 8.7];

  function loadSample() {
    stopAuto();
    if (state.dim === "1d") {
      state.pts1d = BAI_HAT.map(function (v) {
        return { x: v, jitter: Math.random() };
      });
      resetModel("Đã nạp điểm sôi động của 20 bài hát (data/BaiHat.csv).");
    } else {
      var blobs = [
        { x: 3.0, y: 7.0, s: 0.7, n: 60 },
        { x: 6.8, y: 6.6, s: 1.1, n: 80 },
        { x: 5.0, y: 2.8, s: 0.8, n: 60 }
      ];
      state.pts2d = [];
      blobs.forEach(function (b) {
        for (var i = 0; i < b.n; i++) {
          state.pts2d.push({
            x: clamp(b.x + randn() * b.s, 0, DOMAIN),
            y: clamp(b.y + randn() * b.s, 0, DOMAIN)
          });
        }
      });
      resetModel("Đã nạp 200 điểm mẫu từ 3 cụm.");
    }
    render();
  }

  // ------------------------------------------------------------------
  // Thuật toán EM
  // ------------------------------------------------------------------
  function resetModel(msg) {
    state.model = null;
    state.resp = null;
    state.iter = 0;
    state.ll = null;
    state.prevLl = null;
    state.converged = false;
    state.message = msg || "";
  }

  function initModel() {
    stopAuto();
    var pts = points();
    if (pts.length < state.k) {
      state.message = "Cần ít nhất " + state.k + " điểm dữ liệu.";
      render();
      return;
    }
    // Chọn tâm khởi tạo theo kiểu k-means++ (điểm càng xa tâm đã chọn càng dễ được chọn)
    var chosen = [pts[Math.floor(Math.random() * pts.length)]];
    while (chosen.length < state.k) {
      var d2 = pts.map(function (p) {
        var best = Infinity;
        chosen.forEach(function (c) {
          var dx = p.x - c.x;
          var dy = state.dim === "2d" ? p.y - c.y : 0;
          best = Math.min(best, dx * dx + dy * dy);
        });
        return best;
      });
      var total = d2.reduce(function (a, b) { return a + b; }, 0);
      var r = Math.random() * total, acc = 0, idx = 0;
      for (var i = 0; i < pts.length; i++) {
        acc += d2[i];
        if (acc >= r) { idx = i; break; }
      }
      chosen.push(pts[idx]);
    }

    var w = [], mu = [], va = [], cov = [], j;
    var spreadX = variance(pts.map(function (p) { return p.x; }));
    var spreadY = state.dim === "2d"
      ? variance(pts.map(function (p) { return p.y; })) : 0;

    for (j = 0; j < state.k; j++) {
      w.push(1 / state.k);
      if (state.dim === "1d") {
        mu.push(chosen[j].x);
        va.push(Math.max(spreadX, 0.05));
      } else {
        mu.push([chosen[j].x, chosen[j].y]);
        cov.push({ xx: Math.max(spreadX, 0.05), xy: 0, yy: Math.max(spreadY, 0.05) });
      }
    }
    state.model = state.dim === "1d" ? { w: w, mu: mu, va: va } : { w: w, mu: mu, cov: cov };
    state.iter = 0;
    state.prevLl = null;
    state.converged = false;
    eStep();
    state.message = "Đã khởi tạo " + state.k + " cụm. Bấm “Chạy 1 vòng lặp” để xem EM làm việc.";
    render();
  }

  function variance(values) {
    if (values.length < 2) return 1;
    var m = values.reduce(function (a, b) { return a + b; }, 0) / values.length;
    var s = values.reduce(function (a, b) { return a + (b - m) * (b - m); }, 0);
    return s / (values.length - 1);
  }

  /* Bước E: tính xác suất (trách nhiệm) mỗi điểm thuộc từng cụm */
  function eStep() {
    if (!state.model) return;
    var pts = points(), m = state.model, k = state.k;
    var resp = [], totalLl = 0;
    for (var i = 0; i < pts.length; i++) {
      var logp = [];
      for (var j = 0; j < k; j++) {
        var lw = Math.log(Math.max(m.w[j], 1e-12));
        logp.push(lw + (state.dim === "1d"
          ? logNormal1d(pts[i].x, m.mu[j], m.va[j])
          : logNormal2d(pts[i].x, pts[i].y, m.mu[j], m.cov[j])));
      }
      var lse = logSumExp(logp);
      totalLl += lse;
      var row = [];
      for (var j2 = 0; j2 < k; j2++) row.push(Math.exp(logp[j2] - lse));
      resp.push(row);
    }
    state.resp = resp;
    state.prevLl = state.ll;
    state.ll = pts.length ? totalLl / pts.length : null;
  }

  /* Bước M: cập nhật π, μ, σ/Σ từ các xác suất vừa tính */
  function mStep() {
    if (!state.model || !state.resp) return;
    var pts = points(), m = state.model, k = state.k, n = pts.length;
    for (var j = 0; j < k; j++) {
      var nk = 0, i;
      for (i = 0; i < n; i++) nk += state.resp[i][j];
      if (nk < 1e-9) continue;               // cụm rỗng: giữ nguyên tham số
      m.w[j] = nk / n;

      if (state.dim === "1d") {
        var mu = 0;
        for (i = 0; i < n; i++) mu += state.resp[i][j] * pts[i].x;
        mu /= nk;
        var va = 0;
        for (i = 0; i < n; i++) {
          var d = pts[i].x - mu;
          va += state.resp[i][j] * d * d;
        }
        m.mu[j] = mu;
        m.va[j] = va / nk + REG;
      } else {
        var mx = 0, my = 0;
        for (i = 0; i < n; i++) {
          mx += state.resp[i][j] * pts[i].x;
          my += state.resp[i][j] * pts[i].y;
        }
        mx /= nk; my /= nk;
        var sxx = 0, sxy = 0, syy = 0;
        for (i = 0; i < n; i++) {
          var dx = pts[i].x - mx, dy = pts[i].y - my, r = state.resp[i][j];
          sxx += r * dx * dx;
          sxy += r * dx * dy;
          syy += r * dy * dy;
        }
        m.mu[j] = [mx, my];
        m.cov[j] = { xx: sxx / nk + REG, xy: sxy / nk, yy: syy / nk + REG };
      }
    }
    state.iter++;
  }

  /* Một vòng lặp = bước M (dùng xác suất hiện có) + bước E (tính lại xác suất).
     Hội tụ khi log-likelihood gần như không đổi sau một vòng. */
  function runIteration() {
    if (!state.model) return false;
    if (!state.resp) eStep();
    var before = state.ll;
    mStep();
    eStep();
    if (before !== null && state.ll !== null && Math.abs(state.ll - before) < 1e-7) {
      state.converged = true;
    }
    return !state.converged;
  }

  function runMany(count) {
    stopAuto();
    if (!state.model) { initModel(); }
    for (var i = 0; i < count; i++) {
      if (!runIteration()) break;
    }
    state.message = state.converged
      ? "Đã hội tụ sau " + state.iter + " vòng lặp (log-likelihood gần như không đổi)."
      : "";
    render();
  }

  function toggleAuto() {
    if (state.autoTimer) { stopAuto(); render(); return; }
    if (!state.model) initModel();
    if (!state.model) return;
    state.autoTimer = setInterval(function () {
      if (!runIteration()) {
        stopAuto();
        state.message = "Đã hội tụ sau " + state.iter + " vòng lặp.";
      }
      render();
    }, 220);
    render();
  }

  function stopAuto() {
    if (state.autoTimer) { clearInterval(state.autoTimer); state.autoTimer = null; }
  }

  // ------------------------------------------------------------------
  // Vẽ: khung chung
  // ------------------------------------------------------------------
  function setupCanvas() {
    var dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function clear() {
    ctx.setTransform(window.devicePixelRatio || 1, 0, 0, window.devicePixelRatio || 1, 0, 0);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, W, H);
  }

  function pointColor(i) {
    if (!state.resp || !state.resp[i]) return "#8a949c";
    return mixColors(PALETTE.slice(0, state.k), state.resp[i]);
  }

  // --- toạ độ 2D nhìn từ trên xuống ---
  function sx2d(x) { return PAD + (x / DOMAIN) * (W - 2 * PAD); }
  function sy2d(y) { return H - PAD - (y / DOMAIN) * (H - 2 * PAD); }
  function inv2d(px, py) {
    return {
      x: (px - PAD) / (W - 2 * PAD) * DOMAIN,
      y: (H - PAD - py) / (H - 2 * PAD) * DOMAIN
    };
  }

  // --- toạ độ "đèo núi" 2D: các hàng y trượt chéo lên trên ---
  var MTN = { baseY: H - 90, rowOffset: 300, shearX: 150, left: 40, width: W - 230 };
  function sxMtn(x, y) { return MTN.left + (x / DOMAIN) * MTN.width + (y / DOMAIN) * MTN.shearX; }
  function syMtn(y, z, zScale) { return MTN.baseY - (y / DOMAIN) * MTN.rowOffset - z * zScale; }
  function invMtn(px, py) {
    var y = (MTN.baseY - py) / MTN.rowOffset * DOMAIN;
    var x = (px - MTN.left - (y / DOMAIN) * MTN.shearX) / MTN.width * DOMAIN;
    return { x: x, y: y };
  }

  // --- toạ độ 1D ---
  var ONE = { baseY: H - 120, top: 80 };
  function sx1d(x) { return PAD + (x / DOMAIN) * (W - 2 * PAD); }
  function inv1d(px) { return (px - PAD) / (W - 2 * PAD) * DOMAIN; }

  function drawAxisLabel(text, x, y, align) {
    ctx.fillStyle = "#6b7a88";
    ctx.font = "13px 'Segoe UI', Arial, sans-serif";
    ctx.textAlign = align || "center";
    ctx.fillText(text, x, y);
    ctx.textAlign = "left";
  }

  // ------------------------------------------------------------------
  // Vẽ: 2D - khoanh vùng ellipse
  // ------------------------------------------------------------------
  function drawGrid2d() {
    ctx.strokeStyle = "#eceaea";
    ctx.lineWidth = 1;
    for (var i = 0; i <= 10; i++) {
      var t = i / 10;
      var px = PAD + t * (W - 2 * PAD);
      var py = PAD + t * (H - 2 * PAD);
      ctx.beginPath(); ctx.moveTo(px, PAD); ctx.lineTo(px, H - PAD); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(PAD, py); ctx.lineTo(W - PAD, py); ctx.stroke();
    }
    ctx.strokeStyle = "#cfd6db";
    ctx.strokeRect(PAD, PAD, W - 2 * PAD, H - 2 * PAD);
    drawAxisLabel("trục X", W / 2, H - PAD + 26);
    ctx.save();
    ctx.translate(PAD - 28, H / 2);
    ctx.rotate(-Math.PI / 2);
    drawAxisLabel("trục Y", 0, 0);
    ctx.restore();
  }

  function drawEllipse2d() {
    drawGrid2d();
    ctx.save();
    ctx.beginPath();
    ctx.rect(PAD, PAD, W - 2 * PAD, H - 2 * PAD);
    ctx.clip();
    var pts = state.pts2d, i;
    for (i = 0; i < pts.length; i++) {
      ctx.beginPath();
      ctx.arc(sx2d(pts[i].x), sy2d(pts[i].y), 4.5, 0, 2 * Math.PI);
      ctx.fillStyle = pointColor(i);
      ctx.fill();
    }
    if (!state.model) { ctx.restore(); return; }

    var scaleX = (W - 2 * PAD) / DOMAIN;
    var scaleY = (H - 2 * PAD) / DOMAIN;
    for (var j = 0; j < state.k; j++) {
      var cov = state.model.cov[j], mu = state.model.mu[j];
      var e = eigen2d(cov);
      var color = PALETTE[j];
      // vẽ vòng 1σ, 2σ, 3σ
      [1, 2, 3].forEach(function (nSigma, idx) {
        ctx.save();
        ctx.translate(sx2d(mu[0]), sy2d(mu[1]));
        ctx.rotate(-e.angle);                    // trục Y màn hình ngược chiều trục Y dữ liệu
        ctx.beginPath();
        ctx.ellipse(0, 0, Math.sqrt(e.l1) * nSigma * scaleX,
                    Math.sqrt(e.l2) * nSigma * scaleY, 0, 0, 2 * Math.PI);
        ctx.strokeStyle = color;
        ctx.globalAlpha = 0.9 - idx * 0.25;
        ctx.lineWidth = idx === 0 ? 2.5 : 1.5;
        ctx.stroke();
        if (idx === 0) {
          ctx.globalAlpha = 0.12;
          ctx.fillStyle = color;
          ctx.fill();
        }
        ctx.restore();
      });
      // tâm cụm
      var cx = sx2d(mu[0]), cy = sy2d(mu[1]);
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(cx - 7, cy); ctx.lineTo(cx + 7, cy);
      ctx.moveTo(cx, cy - 7); ctx.lineTo(cx, cy + 7);
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.font = "bold 13px 'Segoe UI', Arial, sans-serif";
      ctx.fillText("Cụm " + (j + 1), cx + 10, cy - 10);
    }
    ctx.restore();
  }

  // ------------------------------------------------------------------
  // Vẽ: 2D - đồ thị đèo núi (mặt mật độ dựng 3D giả)
  // ------------------------------------------------------------------
  function drawMountain2d() {
    var m = state.model;
    var N = 60;                                  // độ mịn của lưới
    var grid = [], maxZ = 1e-9, gx, gy;

    for (gy = 0; gy <= N; gy++) {
      var row = [];
      for (gx = 0; gx <= N; gx++) {
        var x = gx / N * DOMAIN, y = gy / N * DOMAIN, z = 0;
        var own = null;
        if (m) {
          own = [];
          for (var j = 0; j < state.k; j++) {
            var d = m.w[j] * normal2d(x, y, m.mu[j], m.cov[j]);
            own.push(d);
            z += d;
          }
          var s = z > 0 ? z : 1;
          own = own.map(function (v) { return v / s; });
        }
        if (z > maxZ) maxZ = z;
        row.push({ x: x, y: y, z: z, own: own });
      }
      grid.push(row);
    }
    var zScale = 300 / maxZ;

    // sàn + lưới toạ độ
    ctx.strokeStyle = "#ece8e8";
    ctx.lineWidth = 1;
    for (var i = 0; i <= 10; i++) {
      var t = i / 10 * DOMAIN;
      ctx.beginPath();
      ctx.moveTo(sxMtn(0, t), syMtn(t, 0, zScale));
      ctx.lineTo(sxMtn(DOMAIN, t), syMtn(t, 0, zScale));
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(sxMtn(t, 0), syMtn(0, 0, zScale));
      ctx.lineTo(sxMtn(t, DOMAIN), syMtn(DOMAIN, 0, zScale));
      ctx.stroke();
    }

    // gom điểm dữ liệu theo hàng để vẽ xen kẽ với các sườn núi (xa vẽ trước)
    var pts = state.pts2d, buckets = [];
    for (i = 0; i <= N; i++) buckets.push([]);
    for (i = 0; i < pts.length; i++) {
      buckets[clamp(Math.round(pts[i].y / DOMAIN * N), 0, N)].push(i);
    }
    function drawBucket(gyIdx) {
      buckets[gyIdx].forEach(function (idx) {
        ctx.beginPath();
        ctx.arc(sxMtn(pts[idx].x, pts[idx].y), syMtn(pts[idx].y, 0, zScale), 3.6, 0, 2 * Math.PI);
        ctx.fillStyle = pointColor(idx);
        ctx.fill();
      });
    }

    if (!m) {
      for (i = N; i >= 0; i--) drawBucket(i);
      drawAxisLabel("Chưa có mô hình — bấm “Khởi tạo cụm”", W / 2, 40);
      return;
    }

    // vẽ từng hàng từ xa tới gần (thuật toán hoạ sĩ) -> tự che khuất như dãy núi
    for (gy = N; gy >= 0; gy--) {
      var r = grid[gy];
      ctx.beginPath();
      ctx.moveTo(sxMtn(r[0].x, r[0].y), syMtn(r[0].y, 0, zScale));
      for (gx = 0; gx <= N; gx++) {
        ctx.lineTo(sxMtn(r[gx].x, r[gx].y), syMtn(r[gx].y, r[gx].z, zScale));
      }
      ctx.lineTo(sxMtn(r[N].x, r[N].y), syMtn(r[N].y, 0, zScale));
      ctx.closePath();

      // màu của sườn núi = pha trộn màu các cụm chiếm ưu thế ở hàng đó
      var mixW = new Array(state.k).fill(0), sumZ = 0;
      for (gx = 0; gx <= N; gx++) {
        sumZ += r[gx].z;
        for (var j2 = 0; j2 < state.k; j2++) mixW[j2] += r[gx].z * r[gx].own[j2];
      }
      if (sumZ > 0) mixW = mixW.map(function (v) { return v / sumZ; });
      else mixW = mixW.map(function () { return 1 / state.k; });

      var rowColor = mixColors(PALETTE.slice(0, state.k), mixW);
      ctx.fillStyle = lighten(rowColor, 0.88);   // đục để che các hàng phía sau
      ctx.fill();
      ctx.strokeStyle = rowColor;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(sxMtn(r[0].x, r[0].y), syMtn(r[0].y, r[0].z, zScale));
      for (gx = 1; gx <= N; gx++) {
        ctx.lineTo(sxMtn(r[gx].x, r[gx].y), syMtn(r[gx].y, r[gx].z, zScale));
      }
      ctx.stroke();
      drawBucket(gy);
    }

    // đỉnh của từng cụm
    for (var j3 = 0; j3 < state.k; j3++) {
      var mu = m.mu[j3];
      var z = 0;
      for (var j4 = 0; j4 < state.k; j4++) {
        z += m.w[j4] * normal2d(mu[0], mu[1], m.mu[j4], m.cov[j4]);
      }
      var px = sxMtn(mu[0], mu[1]), py = syMtn(mu[1], z, zScale);
      ctx.fillStyle = PALETTE[j3];
      ctx.font = "bold 13px 'Segoe UI', Arial, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Cụm " + (j3 + 1), px, py - 10);
      ctx.textAlign = "left";
    }

    drawAxisLabel("trục X", sxMtn(DOMAIN / 2, 0), MTN.baseY + 32);
    drawAxisLabel("trục Y ↗", W - 12, syMtn(DOMAIN, 0, zScale) - 14, "right");
    drawAxisLabel("độ cao = mật độ xác suất của GMM", W / 2, 28);
  }

  // ------------------------------------------------------------------
  // Vẽ: 1D
  // ------------------------------------------------------------------
  function densities1d(x) {
    var m = state.model, out = [], total = 0;
    for (var j = 0; j < state.k; j++) {
      var d = m.w[j] * normal1d(x, m.mu[j], m.va[j]);
      out.push(d);
      total += d;
    }
    return { comp: out, total: total };
  }

  function draw1d() {
    var i, j, gx;
    var baseY = ONE.baseY;

    // trục ngang + lưới dọc
    ctx.strokeStyle = "#eceaea";
    ctx.lineWidth = 1;
    for (i = 0; i <= 10; i++) {
      var px = sx1d(i);
      ctx.beginPath(); ctx.moveTo(px, ONE.top - 20); ctx.lineTo(px, baseY); ctx.stroke();
    }
    ctx.strokeStyle = "#9aa5ad";
    ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(PAD, baseY); ctx.lineTo(W - PAD, baseY); ctx.stroke();
    ctx.fillStyle = "#6b7a88";
    ctx.font = "12px 'Segoe UI', Arial, sans-serif";
    ctx.textAlign = "center";
    for (i = 0; i <= 10; i++) ctx.fillText(String(i), sx1d(i), baseY + 18);
    ctx.textAlign = "left";
    drawAxisLabel("giá trị X — mỗi chấm tròn là một điểm dữ liệu", W / 2, H - 42);

    var STEPS = 400, gridX = [], k = state.k;
    var maxTotal = 1e-9;
    if (state.model) {
      for (gx = 0; gx <= STEPS; gx++) {
        var x = gx / STEPS * DOMAIN;
        var d = densities1d(x);
        gridX.push({ x: x, comp: d.comp, total: d.total });
        if (d.total > maxTotal) maxTotal = d.total;
      }
    }
    var plotH = baseY - ONE.top;
    var zScale = plotH / maxTotal;

    if (state.model && state.view === "mountain") {
      // --- dạng "đèo núi": từng chuông tô nhạt + đường tổng đậm ---
      for (j = 0; j < k; j++) {
        ctx.beginPath();
        ctx.moveTo(sx1d(0), baseY);
        for (gx = 0; gx <= STEPS; gx++) {
          ctx.lineTo(sx1d(gridX[gx].x), baseY - gridX[gx].comp[j] * zScale);
        }
        ctx.lineTo(sx1d(DOMAIN), baseY);
        ctx.closePath();
        ctx.fillStyle = PALETTE[j];
        ctx.globalAlpha = 0.18;
        ctx.fill();
        ctx.globalAlpha = 1;
        ctx.strokeStyle = PALETTE[j];
        ctx.lineWidth = 1.8;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        for (gx = 0; gx <= STEPS; gx++) {
          var py = baseY - gridX[gx].comp[j] * zScale;
          if (gx === 0) ctx.moveTo(sx1d(gridX[gx].x), py);
          else ctx.lineTo(sx1d(gridX[gx].x), py);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.strokeStyle = "#7c3aad";
      ctx.lineWidth = 3;
      ctx.beginPath();
      for (gx = 0; gx <= STEPS; gx++) {
        var pyT = baseY - gridX[gx].total * zScale;
        if (gx === 0) ctx.moveTo(sx1d(gridX[gx].x), pyT);
        else ctx.lineTo(sx1d(gridX[gx].x), pyT);
      }
      ctx.stroke();
      drawAxisLabel("đường tím = mật độ GMM (tổng các chuông) · đường nét đứt = từng chuông",
                    W / 2, 28);
    } else if (state.model) {
      // --- dạng "khoanh vùng": dải μ ± 1σ và μ ± 2σ của từng cụm ---
      // đường mật độ GMM mờ phía sau để đối chiếu
      ctx.strokeStyle = "#cfc6d6";
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (gx = 0; gx <= STEPS; gx++) {
        var pyRef = baseY - gridX[gx].total * zScale * 0.55;
        if (gx === 0) ctx.moveTo(sx1d(gridX[gx].x), pyRef);
        else ctx.lineTo(sx1d(gridX[gx].x), pyRef);
      }
      ctx.stroke();

      var slotH = Math.min(86, (plotH - 60) / k);
      for (j = 0; j < k; j++) {
        var mu = state.model.mu[j], sd = Math.sqrt(state.model.va[j]);
        var barH = Math.max(16, slotH * 0.5);
        var yTop = baseY - 34 - (j + 1) * slotH + (slotH - barH);
        var color = PALETTE[j];
        var x2lo = sx1d(clamp(mu - 2 * sd, 0, DOMAIN));
        var x2hi = sx1d(clamp(mu + 2 * sd, 0, DOMAIN));
        var x1lo = sx1d(clamp(mu - sd, 0, DOMAIN));
        var x1hi = sx1d(clamp(mu + sd, 0, DOMAIN));
        // dải 2σ rồi dải 1σ
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.14;
        ctx.fillRect(x2lo, yTop, x2hi - x2lo, barH);
        ctx.globalAlpha = 0.32;
        ctx.fillRect(x1lo, yTop, x1hi - x1lo, barH);
        ctx.globalAlpha = 1;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x2lo + 0.5, yTop + 0.5, x2hi - x2lo, barH);
        // vạch tâm μ kéo xuống trục dữ liệu
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(sx1d(mu), yTop);
        ctx.lineTo(sx1d(mu), baseY + 6);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = color;
        ctx.font = "bold 12.5px 'Segoe UI', Arial, sans-serif";
        ctx.fillText("Cụm " + (j + 1) + "   μ=" + mu.toFixed(2) + "   σ=" + sd.toFixed(2) +
                     "   π=" + (state.model.w[j] * 100).toFixed(0) + "%", x2lo + 6, yTop - 7);
      }
      drawAxisLabel("mỗi dải = vùng μ ± 1σ (đậm) và μ ± 2σ (nhạt) của một cụm · đường xám = mật độ GMM", W / 2, 28);
    }

    // điểm dữ liệu trên trục
    var pts = state.pts1d;
    for (i = 0; i < pts.length; i++) {
      var yy = baseY + 12 + pts[i].jitter * 16;
      ctx.beginPath();
      ctx.arc(sx1d(pts[i].x), yy, 4.5, 0, 2 * Math.PI);
      ctx.fillStyle = pointColor(i);
      ctx.fill();
      ctx.strokeStyle = "rgba(0,0,0,0.25)";
      ctx.lineWidth = 0.8;
      ctx.stroke();
    }
  }

  // ------------------------------------------------------------------
  // Vẽ: điều phối + bảng tham số
  // ------------------------------------------------------------------
  function render() {
    clear();
    ctx.save();
    if (state.dim === "1d") draw1d();
    else if (state.view === "ellipse") drawEllipse2d();
    else drawMountain2d();
    ctx.restore();
    updatePanel();
  }

  function fmt(v, digits) {
    return (v === null || v === undefined || isNaN(v)) ? "—" : v.toFixed(digits === undefined ? 3 : digits);
  }

  function updatePanel() {
    document.getElementById("stat-n").textContent = points().length;
    document.getElementById("stat-iter").textContent = state.iter;
    document.getElementById("stat-ll").textContent = state.ll === null ? "—" : fmt(state.ll, 4);
    document.getElementById("message").textContent = state.message;
    document.getElementById("btn-auto").textContent =
      state.autoTimer ? "⏸ Dừng" : "▶ Tự động chạy";

    var host = document.getElementById("params");
    if (!state.model) {
      host.innerHTML = "<p class='note'>Chưa có mô hình. Thêm dữ liệu rồi bấm " +
        "<b>Khởi tạo cụm</b>.</p>";
      return;
    }
    var html = "<table><thead><tr><th>Cụm</th><th>π</th>";
    html += state.dim === "1d"
      ? "<th>μ</th><th>σ</th>"
      : "<th>μx</th><th>μy</th><th>σx</th><th>σy</th>";
    html += "</tr></thead><tbody>";
    for (var j = 0; j < state.k; j++) {
      html += "<tr><td><span class='swatch' style='background:" + PALETTE[j] + "'></span>" +
              (j + 1) + "</td><td>" + fmt(state.model.w[j], 3) + "</td>";
      if (state.dim === "1d") {
        html += "<td>" + fmt(state.model.mu[j], 2) + "</td><td>" +
                fmt(Math.sqrt(state.model.va[j]), 2) + "</td>";
      } else {
        html += "<td>" + fmt(state.model.mu[j][0], 2) + "</td><td>" +
                fmt(state.model.mu[j][1], 2) + "</td><td>" +
                fmt(Math.sqrt(state.model.cov[j].xx), 2) + "</td><td>" +
                fmt(Math.sqrt(state.model.cov[j].yy), 2) + "</td>";
      }
      html += "</tr>";
    }
    html += "</tbody></table>";
    host.innerHTML = html;
  }

  function updateHints() {
    var hint = document.getElementById("hint");
    var note = document.getElementById("view-note");
    if (state.dim === "1d") {
      hint.textContent = "Nhấp vào vùng biểu đồ để thêm điểm dữ liệu quanh vị trí trục X đó.";
      note.textContent = state.view === "ellipse"
        ? "Với dữ liệu 1D, “khoanh vùng” hiển thị mỗi cụm bằng dải μ ± 1σ và μ ± 2σ."
        : "Với dữ liệu 1D, “đèo núi” là các đường chuông của từng cụm cùng đường mật độ GMM.";
    } else {
      hint.textContent = state.view === "ellipse"
        ? "Nhấp vào lưới để thêm điểm dữ liệu quanh vị trí đó."
        : "Nhấp lên mặt sàn để thêm điểm; độ cao của núi chính là mật độ xác suất của GMM.";
      note.textContent = state.view === "ellipse"
        ? "Mỗi cụm là 3 vòng ellipse ứng với 1σ, 2σ, 3σ — hướng và độ dẹt lấy từ ma trận hiệp phương sai."
        : "Mặt mật độ của GMM dựng 3D giả: hai cụm chồng lấn sẽ nối nhau thành một “đèo” ở giữa.";
    }
  }

  // ------------------------------------------------------------------
  // Sự kiện
  // ------------------------------------------------------------------
  canvas.addEventListener("click", function (ev) {
    var rect = canvas.getBoundingClientRect();
    var px = (ev.clientX - rect.left) * (W / rect.width);
    var py = (ev.clientY - rect.top) * (H / rect.height);
    var count = parseInt(document.getElementById("points-per-click").value, 10);
    var spread = parseInt(document.getElementById("spread").value, 10) / 100 * 2;

    if (state.dim === "1d") {
      addPoints(inv1d(px), 0, count, spread);
    } else {
      var p = state.view === "ellipse" ? inv2d(px, py) : invMtn(px, py);
      if (p.x < -0.5 || p.x > DOMAIN + 0.5 || p.y < -0.5 || p.y > DOMAIN + 0.5) return;
      addPoints(clamp(p.x, 0, DOMAIN), clamp(p.y, 0, DOMAIN), count, spread);
    }
    render();
  });

  function bindSegment(id, key, onChange) {
    var box = document.getElementById(id);
    box.addEventListener("click", function (ev) {
      var btn = ev.target.closest("button");
      if (!btn) return;
      Array.prototype.forEach.call(box.querySelectorAll("button"), function (b) {
        b.classList.toggle("active", b === btn);
      });
      state[key] = btn.dataset.value;
      if (onChange) onChange();
      updateHints();
      render();
    });
  }

  bindSegment("seg-dim", "dim", function () {
    stopAuto();
    resetModel("Đã chuyển chế độ dữ liệu — hãy khởi tạo lại cụm.");
  });
  bindSegment("seg-view", "view");

  document.getElementById("n-clusters").addEventListener("change", function (e) {
    stopAuto();
    state.k = parseInt(e.target.value, 10);
    resetModel("Đã đổi số cụm — hãy khởi tạo lại.");
    render();
  });

  document.getElementById("btn-init").addEventListener("click", initModel);
  document.getElementById("btn-e").addEventListener("click", function () {
    stopAuto();
    if (!state.model) { initModel(); return; }
    eStep();
    state.message = "Bước E: đã tính lại xác suất thuộc cụm cho từng điểm (màu chấm đổi theo).";
    render();
  });
  document.getElementById("btn-m").addEventListener("click", function () {
    stopAuto();
    if (!state.model) { initModel(); return; }
    mStep();
    state.message = "Bước M: đã cập nhật π, μ và độ lệch của từng cụm từ các xác suất vừa tính.";
    render();
  });
  document.getElementById("btn-1").addEventListener("click", function () { runMany(1); });
  document.getElementById("btn-10").addEventListener("click", function () { runMany(10); });
  document.getElementById("btn-auto").addEventListener("click", toggleAuto);
  document.getElementById("btn-sample").addEventListener("click", loadSample);
  document.getElementById("btn-clear").addEventListener("click", function () {
    stopAuto();
    if (state.dim === "1d") state.pts1d = []; else state.pts2d = [];
    resetModel("Đã xoá toàn bộ dữ liệu.");
    render();
  });

  // ------------------------------------------------------------------
  // Khởi động
  // ------------------------------------------------------------------
  setupCanvas();
  updateHints();
  loadSample();
})();
