
  Following up on the idea of making snapping independent of the currently scrolling node — I gave it a try (passing the node explicitly +
  introducing a snapshot), and ran into a few questions I'd like your input on.

  What I tried

  - Pass the target node (`finished_node` / latched_node) explicitly to SnapAtScrollEnd, instead of reading it from CurrentlyScrollingNode().
  - Introduce a ScrollEndContext snapshot, captured at defer time (scroll direction / input type / scroll axis), to fix the case where B preempts
  the latch and A's snap ends up using B's state.

  Question 1: mid-scroll and scrollend snaps still depend on the currently scrolling node

  The dependency of SnapAtScrollEnd on the currently scrolling node has two layers. The node layer can be decoupled by passing it as a parameter,
  but the context layer (direction / type / scroll axis) is inherently tied to the currently scrolling node in two cases:

  - mid-scroll (a wheel tick's animation finishes mid-gesture and triggers snap): the gesture hasn't ended, so there's no defer and no snapshot —
  the only source is the global gesture state, which belongs to the currently scrolling node.
  - scrollend (the should_snap branch of ScrollEnd): the target node is latched_node (which is the currently scrolling node), and so is the
  context.

  So the node can be decoupled, but the context can't (and doesn't need to) be fully detached in these two cases.

  Question 2: should mid-gesture snap be kept?

  If we dropped mid-gesture snap (i.e. only snap at scrollend, never mid-scroll), full decoupling would be possible — but it would require
  building a context (snapshot) one more time at the scrollend snap, which is actually unnecessary (the global context is already correct at
  scrollend, so we can just read it directly).

  On the other hand, the W3C spec requires snapping at the termination of a scroll (when there's no active scrolling operation), but it does not
  forbid snapping mid-scroll — so mid-scroll snap isn't non-conformant; it's within the UA's latitude.

  So it's a bit contradictory: keeping mid-gesture snap prevents full decoupling; dropping it enables decoupling but that extra "build a context"
  is redundant; and the spec doesn't forbid either. I'd like your take on whether mid-gesture snap should be kept.

  Question 3: a related issue — deferral with no snap can "eat" the preempting node

  Scenario:
  1. A is smooth-scrolling; the GSE arrives while the animation is still running → ScrollEnd defers (A stays latched).
  2. B (a different input type) preempts and latches; CurrentlyScrollingNode() becomes B.
  3. A's animation finishes, but A has no snap container → SnapAtScrollEnd returns false.
  4. It falls through to the deferred delivery ScrollEnd(should_snap=false), which uses CurrentlyScrollingNode() = B as the cleanup target —
  clearing B's latch and prematurely ending B's gesture.

  This shares the same fix direction as the decoupling (the deferred delivery should pass finished_node explicitly rather than relying on
  CurrentlyScrollingNode()), but it's also very low-probability.

  Summary of my questions

  1. Given that full decoupling is impossible while keeping mid-gesture snap, is this optimization still worth doing?
  2. Should mid-gesture snap be kept? (see Question 2)
  3. Should the "deferral with no snap eats the preempting node" issue in Question 3 be fixed separately?

  Best wishes,
  Sanfeng