// A disclosure action menu closes after choosing an action or pressing Escape.
document.addEventListener('click', event => {
  // Mantine renders select options in a portal outside the settings disclosure.
  if (!event.target.closest('.mantine-Combobox-dropdown')) {
    document.querySelectorAll('.advanced-controls[open]').forEach(settings => {
      if (!settings.contains(event.target)) settings.open = false;
    });
  }
  const menu = document.getElementById('file-actions');
  if (!menu || !menu.open) return;
  if (!menu.contains(event.target) || event.target.closest('button')) menu.open = false;
});
document.addEventListener('keydown', event => {
  if (event.key !== 'Escape') return;
  // Let an open select consume Escape before closing its containing settings.
  if (event.target.closest('.advanced-controls')?.querySelector('[aria-expanded="true"]')) return;
  document.querySelectorAll('.advanced-controls[open]').forEach(settings => {
    settings.open = false;
    settings.querySelector('summary')?.focus();
  });
  const menu = document.getElementById('file-actions');
  if (menu?.open) {
    menu.open = false;
    menu.querySelector('summary')?.focus();
  }
});

// Plotly's locked WebGL camera can omit plotly_click. Keep a deliberate click
// selectable using the channel ID from its own picking/hover event, never pixels.
(() => {
  const wired = new WeakSet();
  let scheduled = false;
  function wire() {
    scheduled = false;
    for (const id of ['quality-3d', 'maintenance-3d', 'pack-3d']) {
      const host = document.getElementById(id);
      const plot = host?.querySelector('.js-plotly-plot');
      if (!plot?.on || wired.has(plot)) continue;
      wired.add(plot);
      let picked = null;
      let down = null;
      plot.on('plotly_hover', event => {
        picked = event.points?.find(point => Array.isArray(point.customdata) && point.customdata[0])?.customdata ?? null;
      });
      plot.on('plotly_unhover', () => { picked = null; });
      plot.addEventListener('pointerdown', event => { down = [event.clientX, event.clientY]; });
      plot.addEventListener('pointerup', event => {
        if (!down || Math.hypot(event.clientX-down[0], event.clientY-down[1]) > 5) { down = null; return; }
        down = null;
        requestAnimationFrame(() => {
          if (picked && host.isConnected) window.dash_clientside?.set_props(id, {
            clickData: { points: [{ customdata: picked }], selectionTime: Date.now() }
          });
        });
      });
    }
  }
  new MutationObserver(() => {
    if (!scheduled) { scheduled = true; requestAnimationFrame(wire); }
  }).observe(document.documentElement, { childList:true, subtree:true });
  wire();
})();
