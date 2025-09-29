// Gesture Recognition Module
// Detects a pinch (thumb tip + index finger tip touching/holding) and triggers the Play button.

(function(){
  const MODULE_TAG = '[GestureModule]';

  // MediaPipe landmark indices (Hands):
  // 4 = Thumb tip, 8 = Index finger tip
  const THUMB_TIP = 4;
  const INDEX_TIP = 8;

  // Configurable thresholds
  const PINCH_DISTANCE_THRESHOLD = 0.05; // Normalized distance threshold (tweak if needed)
  const HOLD_DURATION_MS = 2000; // Must hold pinch for this long to trigger action
  const COOLDOWN_MS = 1500; 

  let pinchStartTime = null;
  let inPinch = false;
  let lastTriggerTime = 0;
  let lastLogTime = 0;

  function distance3D(a, b){
    if (!a || !b) return Infinity;
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    const dz = (a.z || 0) - (b.z || 0);
    return Math.sqrt(dx*dx + dy*dy + dz*dz);
  }

  function handleHandResults(e){
    const detail = e.detail || {};
    const hands = detail.multiHandLandmarks || [];

    if (!hands.length){
      inPinch = false;
      pinchStartTime = null;
      return;
    }

    // For now use the first detected hand only
    const hand = hands[0];
    const thumbTip = hand[THUMB_TIP];
    const indexTip = hand[INDEX_TIP];

    if (!thumbTip || !indexTip){
      // Landmarks missing
      return;
    }

    const dist = distance3D(thumbTip, indexTip);

    // Throttle distance logging to avoid spam (every 250ms)
    const now = performance.now();
    if (now - lastLogTime > 250){
      lastLogTime = now; // retain throttling var for potential future use
    }

    if (dist < PINCH_DISTANCE_THRESHOLD){
      if (!inPinch){
        inPinch = true;
        pinchStartTime = now;
      } else {
        // Already in pinch; check hold duration
        const heldFor = now - pinchStartTime;
        if (heldFor >= HOLD_DURATION_MS){
          // Check cooldown
          if (now - lastTriggerTime >= COOLDOWN_MS){
            triggerPlayButton();
            lastTriggerTime = now;
            // Keep pinch state so user must release to trigger again
          }
        }
      }
    } else {
      // Not pinched anymore -> reset state
      inPinch = false;
      pinchStartTime = null;
    }
  }

  function triggerPlayButton(){
    const btn = document.getElementById('playBtn');
    if (!btn){
      return;
    }
    btn.click();
  }

  function init(){
    window.addEventListener('aiHandsResults', handleHandResults);
  }

  // Auto-init on DOMContentLoaded (in case script added late)
  if (document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
