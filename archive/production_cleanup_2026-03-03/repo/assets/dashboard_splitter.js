(function () {
  function wireCmdTabCloseIcons() {
    // Intentionally disabled: close action is controlled by toolbar "×" button.
  }

  function bindSplitter() {
    const splitPane = document.getElementById("split-pane");
    const topPane = document.getElementById("top-pane");
    const divider = document.getElementById("split-divider");
    if (!splitPane || !topPane || !divider) {
      return;
    }
    if (divider.dataset.bound === "1") {
      return;
    }
    divider.dataset.bound = "1";
    divider.dataset.expand = "0";
    divider.dataset.split = "0";
    const dragHint = document.getElementById("divider-drag-hint");

    let dragging = false;
    const startDrag = (ev) => {
      dragging = true;
      document.body.classList.add("split-dragging");
      if (ev && ev.preventDefault) {
        ev.preventDefault();
      }
    };

    const onMove = (clientY) => {
      if (!dragging) {
        return;
      }
      const rect = splitPane.getBoundingClientRect();
      const minTop = 160;
      const minBottom = 180;
      let topPx = clientY - rect.top;
      topPx = Math.max(minTop, Math.min(topPx, rect.height - minBottom));
      topPane.style.flex = `0 0 ${topPx}px`;
      topPane.style.height = `${topPx}px`;
    };

    divider.addEventListener("mousedown", (ev) => {
      if (ev.target && ev.target.closest && ev.target.closest("button")) {
        return;
      }
      startDrag(ev);
    });

    if (dragHint) {
      dragHint.addEventListener("mousedown", (ev) => {
        startDrag(ev);
      });
    }

    window.addEventListener("mousemove", (ev) => {
      onMove(ev.clientY);
    });

    window.addEventListener("mouseup", () => {
      if (!dragging) {
        return;
      }
      dragging = false;
      document.body.classList.remove("split-dragging");
    });

    divider.addEventListener(
      "touchstart",
      (ev) => {
        if (ev.target && ev.target.closest && ev.target.closest("button")) {
          return;
        }
        startDrag(ev);
      },
      { passive: false },
    );

    if (dragHint) {
      dragHint.addEventListener(
        "touchstart",
        (ev) => {
          startDrag(ev);
        },
        { passive: false },
      );
    }

    window.addEventListener(
      "touchmove",
      (ev) => {
        if (!dragging || !ev.touches || !ev.touches.length) {
          return;
        }
        onMove(ev.touches[0].clientY);
        ev.preventDefault();
      },
      { passive: false },
    );

    window.addEventListener("touchend", () => {
      if (!dragging) {
        return;
      }
      dragging = false;
      document.body.classList.remove("split-dragging");
    });

    const splitBtn = document.getElementById("term-divider-split-btn");
    const expandBtn = document.getElementById("term-divider-expand-btn");

    if (splitBtn && splitBtn.dataset.bound !== "1") {
      splitBtn.dataset.bound = "1";
      splitBtn.addEventListener("click", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        const state = Number(divider.dataset.split || "0");
        const next = (state + 1) % 3;
        divider.dataset.split = String(next);
        const ratios = [56, 50, 42];
        const topPx = Math.round((splitPane.clientHeight * ratios[next]) / 100);
        topPane.style.flex = `0 0 ${topPx}px`;
        topPane.style.height = `${topPx}px`;
      });
    }

    if (expandBtn && expandBtn.dataset.bound !== "1") {
      expandBtn.dataset.bound = "1";
      expandBtn.addEventListener("click", (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        const expanded = divider.dataset.expand === "1";
        if (!expanded) {
          divider.dataset.prevTop = topPane.style.height || "";
          const topPx = Math.max(80, Math.round(splitPane.clientHeight * 0.12));
          topPane.style.flex = `0 0 ${topPx}px`;
          topPane.style.height = `${topPx}px`;
          divider.dataset.expand = "1";
        } else {
          const prev = divider.dataset.prevTop || "";
          if (prev) {
            topPane.style.flex = `0 0 ${prev}`;
            topPane.style.height = prev;
          } else {
            const topPx = Math.round(splitPane.clientHeight * 0.56);
            topPane.style.flex = `0 0 ${topPx}px`;
            topPane.style.height = `${topPx}px`;
          }
          divider.dataset.expand = "0";
        }
      });
    }
  }

  const scheduleBind = () => {
    window.requestAnimationFrame(() => {
      bindSplitter();
      wireCmdTabCloseIcons();
    });
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scheduleBind);
  } else {
    scheduleBind();
  }

  const obs = new MutationObserver(() => {
    bindSplitter();
    wireCmdTabCloseIcons();
  });
  obs.observe(document.documentElement, { childList: true, subtree: true });
})();
