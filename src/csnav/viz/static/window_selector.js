/*
 * Window selector: a Leaflet control for isolating trajectory windows.
 *
 * A trajectory's manifest windows overlap by design - adjacent windows share a
 * boundary and each is round-capped - so drawing them all at once in one colour
 * produces a chain of blobs rather than a readable sequence. This control puts
 * each window on its own layer and gives you a tree to pick from: expand a
 * trajectory, tick the windows you want, or "solo" one to see it alone.
 *
 * Layers are addressed as a (window x category) matrix. A layer is on the map
 * when its window is selected AND its category is enabled, so "show every
 * window's roads but no footprints" and "show only window 3, everything about
 * it" are both reachable.
 *
 * The state half of this file is deliberately free of DOM and Leaflet calls so
 * it can be unit-tested under node; see tests/viz/test_window_selector.js.
 */

/**
 * Selection state for a window-selector spec.
 *
 * spec: {categories: [{key, label, enabled}], groups: [{id, label, color,
 *        windows: [{id, label, detail, layers: {categoryKey: <leaflet layer>}}]}]}
 */
function csnavWindowSelectorState(spec) {
  var windowOn = {};
  var categoryOn = {};
  var groups = spec.groups || [];
  var categories = spec.categories || [];

  groups.forEach(function (group) {
    (group.windows || []).forEach(function (win) {
      windowOn[win.id] = true;
    });
  });
  categories.forEach(function (category) {
    categoryOn[category.key] = category.enabled !== false;
  });

  function windowsOf(groupId) {
    var found = groups.filter(function (group) {
      return group.id === groupId;
    })[0];
    return found ? found.windows || [] : [];
  }

  function allWindows() {
    return groups.reduce(function (acc, group) {
      return acc.concat(group.windows || []);
    }, []);
  }

  return {
    /** Whether one window's layers should be drawn at all. */
    isWindowOn: function (windowId) {
      return windowOn[windowId] === true;
    },
    /** Whether a category is enabled across every window. */
    isCategoryOn: function (categoryKey) {
      return categoryOn[categoryKey] === true;
    },
    /** Whether the layer for (window, category) belongs on the map right now. */
    isVisible: function (windowId, categoryKey) {
      return windowOn[windowId] === true && categoryOn[categoryKey] === true;
    },
    setWindow: function (windowId, on) {
      windowOn[windowId] = on === true;
    },
    setCategory: function (categoryKey, on) {
      categoryOn[categoryKey] = on === true;
    },
    /** Turn every window in one group on or off. */
    setGroup: function (groupId, on) {
      windowsOf(groupId).forEach(function (win) {
        windowOn[win.id] = on === true;
      });
    },
    /** Show one window and nothing else - the isolate action. */
    solo: function (windowId) {
      allWindows().forEach(function (win) {
        windowOn[win.id] = win.id === windowId;
      });
    },
    setAll: function (on) {
      allWindows().forEach(function (win) {
        windowOn[win.id] = on === true;
      });
    },
    /** How many of a group's windows are currently on. */
    countOn: function (groupId) {
      return windowsOf(groupId).filter(function (win) {
        return windowOn[win.id] === true;
      }).length;
    },
    groups: groups,
    categories: categories
  };
}

/* Rendering below this line: DOM and Leaflet only. */

function csnavWindowSelector(map, spec, options) {
  var settings = options || {};
  var state = csnavWindowSelectorState(spec);
  var expanded = {};
  var rendered = null;

  function element(tag, className, text) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined) {
      node.textContent = text;
    }
    return node;
  }

  function applyLayers() {
    state.groups.forEach(function (group) {
      (group.windows || []).forEach(function (win) {
        Object.keys(win.layers || {}).forEach(function (key) {
          var layer = win.layers[key];
          if (!layer) {
            return;
          }
          if (state.isVisible(win.id, key)) {
            if (!map.hasLayer(layer)) {
              map.addLayer(layer);
            }
          } else if (map.hasLayer(layer)) {
            map.removeLayer(layer);
          }
        });
      });
    });
  }

  function refresh() {
    applyLayers();
    if (rendered) {
      rendered();
    }
  }

  function buildCategories(container) {
    if (state.categories.length < 2) {
      return function () {};
    }
    var row = element('div', 'csnav-ws-cats');
    var inputs = [];
    state.categories.forEach(function (category) {
      var label = element('label', 'csnav-ws-cat');
      var box = document.createElement('input');
      box.type = 'checkbox';
      box.checked = state.isCategoryOn(category.key);
      box.addEventListener('change', function () {
        state.setCategory(category.key, box.checked);
        refresh();
      });
      label.appendChild(box);
      label.appendChild(element('span', null, category.label));
      row.appendChild(label);
      inputs.push({ box: box, key: category.key });
    });
    container.appendChild(row);
    return function () {
      inputs.forEach(function (entry) {
        entry.box.checked = state.isCategoryOn(entry.key);
      });
    };
  }

  function buildGroup(group, container) {
    var block = element('div', 'csnav-ws-group');
    var head = element('div', 'csnav-ws-grouphead');
    var caret = element('span', 'csnav-ws-caret', '▸');
    var swatch = element('span', 'csnav-ws-swatch');
    swatch.style.background = group.color || '#666666';
    var name = element('span', 'csnav-ws-name', group.label || group.id);
    var count = element('span', 'csnav-ws-count');

    head.appendChild(caret);
    head.appendChild(swatch);
    head.appendChild(name);
    head.appendChild(count);

    ['all', 'none'].forEach(function (action) {
      var button = element('button', 'csnav-ws-btn', action);
      button.type = 'button';
      button.title = action === 'all' ? 'show every window' : 'hide every window';
      button.addEventListener('click', function (event) {
        event.stopPropagation();
        state.setGroup(group.id, action === 'all');
        refresh();
      });
      head.appendChild(button);
    });

    var list = element('div', 'csnav-ws-windows');
    var boxes = [];
    (group.windows || []).forEach(function (win) {
      var row = element('div', 'csnav-ws-window');
      var label = element('label', 'csnav-ws-winlabel');
      var box = document.createElement('input');
      box.type = 'checkbox';
      box.checked = state.isWindowOn(win.id);
      box.addEventListener('change', function () {
        state.setWindow(win.id, box.checked);
        refresh();
      });
      label.appendChild(box);
      label.appendChild(element('span', 'csnav-ws-wintext', win.label));
      if (win.detail) {
        label.title = win.detail;
      }
      row.appendChild(label);

      var solo = element('button', 'csnav-ws-btn', 'solo');
      solo.type = 'button';
      solo.title = 'show only this window';
      solo.addEventListener('click', function (event) {
        event.stopPropagation();
        state.solo(win.id);
        refresh();
      });
      row.appendChild(solo);
      list.appendChild(row);
      boxes.push({ box: box, id: win.id });
    });

    head.addEventListener('click', function () {
      expanded[group.id] = !expanded[group.id];
      refresh();
    });

    block.appendChild(head);
    block.appendChild(list);
    container.appendChild(block);

    return function () {
      var total = (group.windows || []).length;
      count.textContent = state.countOn(group.id) + '/' + total;
      caret.textContent = expanded[group.id] ? '▾' : '▸';
      list.style.display = expanded[group.id] ? 'block' : 'none';
      boxes.forEach(function (entry) {
        entry.box.checked = state.isWindowOn(entry.id);
      });
    };
  }

  var Control = L.Control.extend({
    onAdd: function () {
      var root = element('div', 'csnav-ws leaflet-bar');
      var head = element('div', 'csnav-ws-head');
      head.appendChild(element('span', 'csnav-ws-title', spec.title || 'Windows'));

      ['all', 'none'].forEach(function (action) {
        var button = element('button', 'csnav-ws-btn', action);
        button.type = 'button';
        button.title = action === 'all' ? 'show every window' : 'hide every window';
        button.addEventListener('click', function () {
          state.setAll(action === 'all');
          refresh();
        });
        head.appendChild(button);
      });
      root.appendChild(head);

      var refreshCategories = buildCategories(root);
      var body = element('div', 'csnav-ws-body');
      var refreshers = state.groups.map(function (group) {
        return buildGroup(group, body);
      });
      root.appendChild(body);

      rendered = function () {
        refreshCategories();
        refreshers.forEach(function (fn) {
          fn();
        });
      };

      L.DomEvent.disableClickPropagation(root);
      L.DomEvent.disableScrollPropagation(root);
      refresh();
      return root;
    }
  });

  var control = new Control({ position: settings.position || 'topright' });
  control.addTo(map);
  if (spec.groups && spec.groups.length === 1) {
    expanded[spec.groups[0].id] = true;
    refresh();
  }
  return control;
}

/* Exported for the node-based unit tests; `module` is undefined in a browser. */
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { csnavWindowSelectorState: csnavWindowSelectorState };
}
