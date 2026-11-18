/** Compact floating YouTube preview for timestamp clicks. Does not touch the map layout. */
import { el, toast } from './shared.js';

const VIDEO_ID = /^[A-Za-z0-9_-]{11}$/;
let root = null;
let frame = null;
let fallback = null;
let currentId = null;
let pointer = { x: 0, y: 0 };

document.addEventListener('pointerdown', (event) => {
  pointer = { x: event.clientX, y: event.clientY };
}, true);

function clamp(value, min, max) {
  return Math.max(min, Math.min(value, max));
}

function place(x, y) {
  const rect = root?.getBoundingClientRect();
  const width = rect?.width || 384;
  const height = rect?.height || 274;
  const left = clamp(x - width / 2, 8, Math.max(8, window.innerWidth - width - 8));
  const top = y + height + 16 > window.innerHeight ? clamp(y - height - 16, 8, window.innerHeight - height - 8)
                                                    : clamp(y + 16, 8, window.innerHeight - height - 8);
  root.style.left = `${left}px`;
  root.style.top = `${top}px`;
}

function makeDraggable(handle) {
  let start = null;
  const move = (event) => {
    if (!start) return;
    const rect = root.getBoundingClientRect();
    root.style.left = `${clamp(rect.left + event.clientX - start.x, 4, Math.max(4, window.innerWidth - rect.width - 4))}px`;
    root.style.top = `${clamp(rect.top + event.clientY - start.y, 4, Math.max(4, window.innerHeight - rect.height - 4))}px`;
    start = { x: event.clientX, y: event.clientY };
  };
  const up = () => {
    start = null;
    window.removeEventListener('pointermove', move);
    window.removeEventListener('pointerup', up);
  };
  handle.addEventListener('pointerdown', (event) => {
    if (event.target.closest('button, a, iframe, .tm-video-preview-resize')) return;
    start = { x: event.clientX, y: event.clientY };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  });
}

function makeResizable(handle, head) {
  let start = null;
  const move = (event) => {
    if (!start) return;
    const rect = root.getBoundingClientRect();
    const headHeight = head?.offsetHeight || 32;
    const minWidth = 260;
    const maxWidth = Math.max(
      minWidth,
      Math.min(960, window.innerWidth - rect.left - 8, Math.floor((window.innerHeight - rect.top - 8 - headHeight) * 16 / 9)),
    );
    const width = clamp(event.clientX - rect.left, minWidth, maxWidth);
    root.style.width = `${width}px`;
    // Keep the resized player inside the viewport.
    const after = root.getBoundingClientRect();
    root.style.left = `${clamp(after.left, 4, Math.max(4, window.innerWidth - after.width - 4))}px`;
    root.style.top = `${clamp(after.top, 4, Math.max(4, window.innerHeight - after.height - 4))}px`;
  };
  const up = () => {
    start = null;
    window.removeEventListener('pointermove', move);
    window.removeEventListener('pointerup', up);
  };
  handle.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    event.stopPropagation();
    start = { x: event.clientX, y: event.clientY };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  });
}

function buildPlayer(videoId) {
  root = el('div', { class: 'tm-video-preview', role: 'dialog', 'aria-label': 'Video preview' });
  const head = el('div', { class: 'tm-video-preview-head' }, [
    el('span', { class: 'tm-video-preview-title' }, 'Video preview'),
    fallback = el('a', {
      class: 'tm-video-preview-link',
      target: '_blank',
      rel: 'noopener',
      title: 'Open this moment on YouTube',
      hidden: true,
    }, 'Open on YouTube'),
    el('span', { class: 'tm-video-preview-resize', title: 'Resize video preview', 'aria-hidden': 'true' }),
    el('button', {
      class: 'tm-video-preview-close',
      type: 'button',
      'aria-label': 'Close video preview',
      title: 'Close',
      onclick: closePreview,
    }, '×'),
  ]);
  const grip = head.querySelector('.tm-video-preview-resize');
  const box = el('div', { class: 'tm-video-preview-frame' }, [
    frame = el('iframe', {
      title: 'YouTube video preview',
      allow: 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share',
      allowfullscreen: true,
      referrerpolicy: 'strict-origin-when-cross-origin',
    }),
  ]);
  root.append(head, box);
  document.body.append(root);
  makeDraggable(head);
  makeResizable(grip, head);
  const watch = setTimeout(() => {
    if (root) {
      fallback.hidden = false;
      root.classList.add('tm-video-preview-fallback');
    }
  }, 8000);
  frame.addEventListener('load', () => clearTimeout(watch), { once: true });
  return root;
}

/** Open or update the single floating player at an exact video moment. */
export function showVideoPreview({ videoId, seconds, title = '', x = pointer.x, y = pointer.y,
                                   backendUrl = 'http://127.0.0.1:8765' }) {
  if (!VIDEO_ID.test(String(videoId || ''))) {
    toast('This map is not linked to a valid YouTube video.');
    return false;
  }
  const at = Math.max(0, Math.floor(Number(seconds) || 0));
  const base = String(backendUrl || 'http://127.0.0.1:8765').replace(/\/+$/, '');
  const playerUrl = new URL(`${base}/player/${videoId}`);
  playerUrl.searchParams.set('t', String(at));
  if (!root) {
    buildPlayer(videoId);
    place(x, y);
    frame.src = playerUrl.toString();
    currentId = videoId;
  } else {
    place(x, y);
    if (currentId !== videoId) {
      frame.src = playerUrl.toString();
      currentId = videoId;
    } else {
      // Same video: seek inside the existing player instead of opening another one.
      try {
        frame.contentWindow?.postMessage({ type: 'mivmap:seek', seconds: at }, new URL(base).origin);
      } catch {
        frame.src = playerUrl.toString();
      }
    }
  }
  fallback.href = `https://www.youtube.com/watch?v=${videoId}&t=${at}s`;
  fallback.hidden = false; // always available if embedding is blocked
  return true;
}

/** Remove the player and stop playback. */
export function closePreview() {
  if (!root) return;
  if (frame) frame.src = 'about:blank';
  root.remove();
  root = null;
  frame = null;
  fallback = null;
  currentId = null;
}

export function hasPreview() {
  return !!root;
}
