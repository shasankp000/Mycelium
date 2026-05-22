// ---------------------------------------------------------------------------
// ClusterNode — custom canvas label overlay for cluster nodes.
// In Phase 4 this is passed as a custom node renderer to ForceGraph2D.
// For now it is a typed stub so imports don't break.
// Plan ref: §2.3.7
// ---------------------------------------------------------------------------

import type { GraphNode } from '../../types/graph';

/**
 * drawClusterNode
 * Called by Phase 4 ForceGraph2D nodeCanvasObject callback.
 * Draws a violet circle + node-count badge.
 */
export function drawClusterNode(
  node:  GraphNode,
  ctx:   CanvasRenderingContext2D,
  scale: number,
): void {
  const r = (node.size ?? 10) / scale;
  const x = (node as unknown as { x: number }).x ?? 0;
  const y = (node as unknown as { y: number }).y ?? 0;

  // Outer ring
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fillStyle   = node.color ?? '#8b5cf6';
  ctx.globalAlpha = node.opacity ?? 0.85;
  ctx.fill();

  // Dashed border to visually mark as expandable
  ctx.setLineDash([2, 2]);
  ctx.strokeStyle = 'rgba(255,255,255,0.5)';
  ctx.lineWidth   = 1 / scale;
  ctx.stroke();
  ctx.setLineDash([]);

  // Badge: node count
  if ((node.clusterNodeCount ?? 0) > 0) {
    ctx.globalAlpha = 1;
    ctx.font        = `${Math.max(8, 10 / scale)}px sans-serif`;
    ctx.fillStyle   = '#ffffff';
    ctx.textAlign   = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(node.clusterNodeCount), x, y);
  }

  ctx.globalAlpha = 1;
}
