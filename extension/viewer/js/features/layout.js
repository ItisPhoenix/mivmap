/**
 * Pure viewer layout decisions, shared by the app shell and the layout tests.
 *
 * Desktop reserves the inspector width so the map is never drawn underneath it.
 * Medium/narrow viewports turn the side panels into dismissible drawers that
 * overlay the canvas without permanently consuming its width.
 */
export const LAYOUT_BREAKPOINTS = { medium: 1100, narrow: 760 };

export function layoutState(width, { outlineOpen = false, inspectorOpen = false } = {}) {
  const mode = width <= LAYOUT_BREAKPOINTS.narrow
    ? 'narrow'
    : width <= LAYOUT_BREAKPOINTS.medium
      ? 'medium'
      : 'desktop';
  const reserveInspector = mode === 'desktop' && inspectorOpen;
  // Medium keeps the outline in flow and only turns the inspector into a drawer;
  // narrow turns both into drawers. The backdrop must only cover an actual drawer.
  const drawersActive = mode === 'medium'
    ? inspectorOpen
    : mode === 'narrow'
      ? outlineOpen || inspectorOpen
      : false;
  return { mode, outlineOpen, inspectorOpen, reserveInspector, drawersActive };
}
