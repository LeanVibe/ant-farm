# PRD: Mobile PWA Dashboard & Push Notifications

## Executive Summary

The Mobile PWA Dashboard is the **human control center** for LeanVibe Agent Hive 2.0. Built with Lit + Vite + Tailwind, it surfaces backlog, sprint focus, agent health, and real-time event streams in a responsive Progressive Web App that installs to home-screen. Push notifications (Firebase Cloud Messaging) ensure founders receive timely alerts—**build success, agent errors, merge requests—**directly on their phone.

## Problem Statement

Developers and product owners lack an at-a-glance view of multi-agent activity. Slack logs and terminal panes are noisy, causing:
- **Missed Alerts**: Critical build failures seen hours later
- **Poor Prioritization**: No unified backlog across agents
- **Limited Mobility**: Cannot monitor while away from laptop
- **On-Call Pain**: Hard to triage which agent is failing

## Success Metrics

| Metric | Target |
|---|---|
|PWA Install Rate|≥70% of invited users by day 7|
|Alert Response Time|<5 min median from push to acknowledge|
|Failed Build Detection|100% surfaced via push|
|Dashboard FPS|≥45 fps on low-end Android| 
|Offline Availability|Core kanban view functional offline| 

## Core Features

### 1. Kanban Backlog & Sprint Board
Visualize Product Manager backlog, WIP, Done columns. Drag-and-drop reorder (indexedDB offline cache, optimistic update). Filters by agent, priority.

### 2. Real-Time Event Stream
WebSocket (wss://api.hive.local/events) pipes Claude Hook events. Timeline view with color-coded agent badges.

### 3. Push Notifications
Firebase Cloud Messaging topics:
- `build.failed` (high priority, sound)
- `agent.error`
- `task.completed`
- `human.approval.request`
**User Story**: As a founder, I receive a push when agents need my decision.

### 4. Agent Health Panel
Sparkline of CPU/token usage, uptime. Green/amber/red status derived from Prometheus.

### 5. Secure Login & RBAC
JWT via Auth0 + refresh, roles: admin, observer. Biometric WebAuthn optional.

## Technical Architecture

```
+-------------+    HTTPS     +---------------+    WebSocket   +----------------+
|  PWA Client |  <------->  |  FastAPI API  |  <----------> |  Event Broker   |
+-------------+             +---------------+               +----------------+
       ^ FCM push                              ^ Redis Streams / NATS
       |                                        |
   Firebase                                    Agent Hooks
```

### Key Tech Choices
- **Framework**: Lit (web components) + TypeScript
- **State**: Zustand (portable, tiny)
- **PWA**: Workbox offline caching, add-to-home-screen prompt
- **Push**: FCM web push; service worker handles background clicks

### Data Shape
```json
{
  "taskId": "123e4567",
  "title": "Implement Redis HA",
  "status": "In-Progress",
  "agent": "backend-1",
  "priority": "High",
  "updatedAt": "2025-07-26T11:32:00Z"
}
```

## Implementation Plan

| Sprint | Deliverables |
|---|---|
|Week 1|Scaffold Vite + Lit PWA, service worker, auth flow|
|Week 2|Kanban board (offline first), WebSocket subscription, unit tests|
|Week 3|Push notification topics, FCM integration, agent health panel|
|Week 4|UX polish, Lighthouse PWA score ≥90, end-to-end Cypress tests|

### Sample Cypress Test
```typescript
it('moves card to Done', () => {
  cy.login('founder');
  cy.contains('Implement Redis HA').drag('[data-column="Done"]');
  cy.contains('Implement Redis HA').should('exist').parents('[data-column]').should('have.attr','data-column','Done');
});
```

## Risks & Mitigation
| Risk | Mitigation |
|---|---|
|Push reliability on iOS|Use APNs-via-FCM; fallback email digest|
|Offline data staleness|Stale-while-revalidate pattern; show sync badge|
|Auth token theft|SameSite=Lax cookies, rotating refresh tokens|

## Dependencies
- Observability API (WebSocket)
- Firebase project keys
- User/RBAC service

## Acceptance Criteria
- PWA installable, offline board loads from cache
- Push notification received within 5 s of event publish
- Dragging backlog item updates server and other clients via WebSocket
- Works on Chrome, Safari, Firefox (mobile & desktop)

## Future Enhancements
- **Voice Commands** (Web Speech API)
- **Dark Mode Auto** (prefers-color-scheme)
- **Widget Embeds** into existing Slack channels