/*
 * Unit tests for the window selector's selection logic, run under node.
 *
 * Only the state half is exercised - it is deliberately free of DOM and
 * Leaflet calls so it can be tested at all. The rendering half is covered
 * from Python by asserting the emitted page wires the right layers up.
 *
 * Run: node tests/viz/test_window_selector.js
 * (tests/viz/test_window_selector.py runs this under pytest.)
 */

var assert = require('assert');
var path = require('path');
var selector = require(
  path.join(__dirname, '..', '..', 'src', 'csnav', 'viz', 'static', 'window_selector.js')
);

function spec() {
  return {
    title: 'Manifest windows',
    categories: [
      { key: 'footprint', label: 'footprints' },
      { key: 'tiles', label: 'tiles', enabled: false }
    ],
    groups: [
      {
        id: 't_p',
        label: 't_p',
        color: '#B2182B',
        windows: [
          { id: 't_p:0000', label: '0000', layers: {} },
          { id: 't_p:0001', label: '0001', layers: {} }
        ]
      },
      {
        id: 't_alt',
        label: 't_alt',
        color: '#0072B2',
        windows: [{ id: 't_alt:0000', label: '0000', layers: {} }]
      }
    ]
  };
}

var tests = {
  'every window starts on': function () {
    var state = selector.csnavWindowSelectorState(spec());
    assert.strictEqual(state.isWindowOn('t_p:0000'), true);
    assert.strictEqual(state.isWindowOn('t_alt:0000'), true);
    assert.strictEqual(state.countOn('t_p'), 2);
  },

  'a category honours its declared initial state': function () {
    var state = selector.csnavWindowSelectorState(spec());
    assert.strictEqual(state.isCategoryOn('footprint'), true);
    assert.strictEqual(state.isCategoryOn('tiles'), false);
  },

  'a layer is visible only when its window and its category are both on': function () {
    var state = selector.csnavWindowSelectorState(spec());
    assert.strictEqual(state.isVisible('t_p:0000', 'footprint'), true);
    assert.strictEqual(state.isVisible('t_p:0000', 'tiles'), false);

    state.setCategory('tiles', true);
    assert.strictEqual(state.isVisible('t_p:0000', 'tiles'), true);

    state.setWindow('t_p:0000', false);
    assert.strictEqual(state.isVisible('t_p:0000', 'tiles'), false);
    assert.strictEqual(state.isVisible('t_p:0001', 'tiles'), true);
  },

  'solo isolates one window across every group': function () {
    var state = selector.csnavWindowSelectorState(spec());
    state.solo('t_p:0001');

    assert.strictEqual(state.isWindowOn('t_p:0001'), true);
    assert.strictEqual(state.isWindowOn('t_p:0000'), false);
    assert.strictEqual(state.isWindowOn('t_alt:0000'), false);
    assert.strictEqual(state.countOn('t_p'), 1);
    assert.strictEqual(state.countOn('t_alt'), 0);
  },

  'solo does not disturb the category mask': function () {
    var state = selector.csnavWindowSelectorState(spec());
    state.setCategory('tiles', true);
    state.solo('t_p:0000');
    assert.strictEqual(state.isCategoryOn('tiles'), true);
    assert.strictEqual(state.isVisible('t_p:0000', 'tiles'), true);
  },

  'setGroup touches only its own group': function () {
    var state = selector.csnavWindowSelectorState(spec());
    state.setGroup('t_p', false);

    assert.strictEqual(state.countOn('t_p'), 0);
    assert.strictEqual(state.countOn('t_alt'), 1);

    state.setGroup('t_p', true);
    assert.strictEqual(state.countOn('t_p'), 2);
  },

  'setAll covers every group': function () {
    var state = selector.csnavWindowSelectorState(spec());
    state.setAll(false);
    assert.strictEqual(state.countOn('t_p') + state.countOn('t_alt'), 0);

    state.setAll(true);
    assert.strictEqual(state.countOn('t_p') + state.countOn('t_alt'), 3);
  },

  'an unknown group counts zero rather than throwing': function () {
    var state = selector.csnavWindowSelectorState(spec());
    assert.strictEqual(state.countOn('nope'), 0);
    state.setGroup('nope', false);
  },

  'an empty spec is harmless': function () {
    var state = selector.csnavWindowSelectorState({});
    state.setAll(true);
    assert.strictEqual(state.isWindowOn('anything'), false);
    assert.strictEqual(state.isVisible('anything', 'footprint'), false);
  }
};

var failures = 0;
Object.keys(tests).forEach(function (name) {
  try {
    tests[name]();
    console.log('ok   - ' + name);
  } catch (error) {
    failures += 1;
    console.log('FAIL - ' + name + '\n       ' + error.message);
  }
});
console.log(Object.keys(tests).length - failures + '/' + Object.keys(tests).length + ' passed');
process.exit(failures === 0 ? 0 : 1);
