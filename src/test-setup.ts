import { afterEach } from "bun:test";
import { cleanup } from "@testing-library/react";
import { Window } from "happy-dom";

const testWindow = new Window({ url: "http://localhost/" });

Object.defineProperties(globalThis, {
  window: { configurable: true, value: testWindow },
  self: { configurable: true, value: testWindow },
  document: { configurable: true, value: testWindow.document },
  navigator: { configurable: true, value: testWindow.navigator },
  location: { configurable: true, value: testWindow.location },
  localStorage: { configurable: true, value: testWindow.localStorage },
  sessionStorage: { configurable: true, value: testWindow.sessionStorage },
});

for (const property of Object.getOwnPropertyNames(testWindow)) {
  if (property in globalThis) continue;
  Object.defineProperty(globalThis, property, {
    configurable: true,
    get: () => Reflect.get(testWindow, property),
  });
}

afterEach(() => {
  cleanup();
  testWindow.localStorage.clear();
  testWindow.sessionStorage.clear();
});
