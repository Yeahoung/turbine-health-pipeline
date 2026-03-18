(function() {
  const slides = Array.from(document.querySelectorAll('.slide'));
  const navDots = document.getElementById('navDots');
  const progressBar = document.getElementById('progressBar');
  const slideMeta = document.getElementById('slideMeta');
  const slideTitle = document.getElementById('slideTitle');
  const fullscreenBtn = document.getElementById('fullscreenBtn');
  const presenterBtn = document.getElementById('presenterBtn');
  const autoBtn = document.getElementById('autoBtn');
  const startBtn = document.getElementById('startBtn');
  const modeChip = document.getElementById('modeChip');
  const presenterSlideName = document.getElementById('presenterSlideName');
  const presenterNotes = document.getElementById('presenterNotes');
  const elapsedChip = document.getElementById('elapsedChip');
  const slideChip = document.getElementById('slideChip');
  const autoChip = document.getElementById('autoChip');
  const architectureFigure = document.getElementById('architectureFigure');
  const cycleArchitectureBtn = document.getElementById('cycleArchitecture');
  const imageModal = document.getElementById('imageModal');
  const modalImage = document.getElementById('modalImage');
  const modalCaption = document.getElementById('modalCaption');
  const modalClose = document.getElementById('modalClose');
  const cueList = document.getElementById('cueList');

  let activeIndex = 0;
  let autoPlay = false;
  let autoTimer = null;
  let highlightTimer = null;
  let activeHighlightIndex = 0;
  const startTime = Date.now();
  const highlightSequence = ['main-1', 'main-2', 'main-3', 'main-4', 'main-5', 'main-6', 'advisory', 'loop'];

  const cuesBySlide = {
    'Title': ['Lead with the system identity.', 'Say advisory-only clearly.', 'Avoid overclaiming autonomy.'],
    'Problem': ['Describe the gap from fault detection to action.', 'Frame the project as software-first.', 'Keep the pain point practical.'],
    'System Overview': ['Explain the three coordinated flows.', 'Main flow first, then RL advisory flow.', 'Reinforce human-in-the-loop governance.'],
    'Pipeline': ['Walk the main flow left to right.', 'Introduce RL second.', 'Call it a governance chain.'],
    'Metadata': ['Use the phrase white-box features.', 'Explain why metadata is auditable.', 'Connect metadata to explainable AI.'],
    'RL Advisory Module': ['Keep RL simulation-based and advisory-only.', 'Mention multi-objective optimization clearly.', 'Do not imply direct turbine actuation.'],
    'Case Study': ['Point out different health states.', 'Show different operator postures.', 'Emphasise what happens next.'],
    'Explainable & Controllable AI': ['Show explainability mechanisms first.', 'Then explain controllability and safety gates.', 'End with operator final authority.'],
    'Engineering Value': ['Cover operations, maintenance, sustainability, and carbon.', 'Keep value claims practical.', 'Repeat system-level contribution.'],
    'Deployment Path': ['Prototype now, scale later.', 'Mention data, simulation, and multi-turbine roadmap.', 'Keep it feasible.'],
    'Conclusion': ['Finish with responsible AI.', 'Repeat advisory-only message.', 'Pause after the final line.'],
    'Thank You': ['Thank the audience.', 'Invite questions.', 'Keep it brief.']
  };

  function getUrlState() {
    const params = new URLSearchParams(window.location.search);
    return {
      presenter: params.get('presenter') === '1',
      autoplay: params.get('autoplay') === '1'
    };
  }

  function setUrlParam(key, enabled) {
    const url = new URL(window.location.href);
    if (enabled) url.searchParams.set(key, '1');
    else url.searchParams.delete(key);
    history.replaceState({}, '', url);
  }

  function scrollToSlide(index) {
    const boundedIndex = Math.max(0, Math.min(index, slides.length - 1));
    slides[boundedIndex].scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function renderDots() {
    if (!navDots) return;
    navDots.innerHTML = slides.map((slide, index) => `
      <button class="nav-dot ${index === activeIndex ? 'active' : ''}" aria-label="${slide.dataset.title}" title="${slide.dataset.title}" data-index="${index}"></button>
    `).join('');

    navDots.querySelectorAll('button').forEach((button) => {
      button.addEventListener('click', () => scrollToSlide(Number(button.dataset.index)));
    });
  }

  function updatePresenter() {
    if (!slides[activeIndex]) return;
    const slide = slides[activeIndex];
    const title = slide.dataset.title || `Slide ${activeIndex + 1}`;
    if (presenterSlideName) presenterSlideName.textContent = title;
    if (presenterNotes) presenterNotes.textContent = slide.dataset.notes || 'No notes available for this slide.';
    if (slideChip) slideChip.textContent = `Slide ${activeIndex + 1} / ${slides.length}`;

    const cues = cuesBySlide[title] || ['Explain the core idea clearly.', 'Keep the message practical.', 'End with a takeaway.'];
    if (cueList) cueList.innerHTML = cues.map(c => `<li>${c}</li>`).join('');
  }

  function updateTopbar() {
    if (!slides[activeIndex]) return;
    const title = slides[activeIndex].dataset.title || `Slide ${activeIndex + 1}`;
    if (slideMeta) slideMeta.textContent = `Slide ${activeIndex + 1} / ${slides.length}`;
    if (slideTitle) slideTitle.textContent = title;
    if (progressBar) progressBar.style.width = `${((activeIndex + 1) / slides.length) * 100}%`;
    if (modeChip) modeChip.textContent = `${document.body.classList.contains('presenter-mode') ? 'Presenter' : 'Presentation'}${autoPlay ? ' • Auto' : ''}`;
    updatePresenter();
  }

  function toggleFullscreen() {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
    else document.exitFullscreen?.();
  }

  function togglePresenter(forceState) {
    const nextState = typeof forceState === 'boolean' ? forceState : !document.body.classList.contains('presenter-mode');
    document.body.classList.toggle('presenter-mode', nextState);
    setUrlParam('presenter', nextState);
    updateTopbar();
  }

  function startAutoPlay() {
    stopAutoPlay();
    autoPlay = true;
    if (autoChip) autoChip.textContent = 'Auto mode on';
    if (autoBtn) autoBtn.textContent = 'Pause Auto';
    setUrlParam('autoplay', true);
    autoTimer = setInterval(() => {
      if (activeIndex >= slides.length - 1) {
        scrollToSlide(0);
      } else {
        scrollToSlide(activeIndex + 1);
      }
    }, 9000);
    updateTopbar();
  }

  function stopAutoPlay() {
    autoPlay = false;
    if (autoChip) autoChip.textContent = 'Manual mode';
    if (autoBtn) autoBtn.textContent = 'Auto Play';
    setUrlParam('autoplay', false);
    clearInterval(autoTimer);
    autoTimer = null;
    updateTopbar();
  }

  function toggleAutoPlay() {
    if (autoPlay) stopAutoPlay();
    else startAutoPlay();
  }

  function setArchitectureHighlight(target) {
    if (architectureFigure) architectureFigure.dataset.highlight = target;
    const stageCards = Array.from(document.querySelectorAll('.stage-card'));
    stageCards.forEach(card => card.classList.toggle('active', card.dataset.highlight === target));
  }

  function cycleArchitecture() {
    clearInterval(highlightTimer);
    let count = 0;
    highlightTimer = setInterval(() => {
      setArchitectureHighlight(highlightSequence[activeHighlightIndex % highlightSequence.length]);
      activeHighlightIndex += 1;
      count += 1;
      if (count >= highlightSequence.length) clearInterval(highlightTimer);
    }, 1100);
  }

  function openModal(src, caption, alt) {
    if (modalImage) modalImage.src = src;
    if (modalImage) modalImage.alt = alt || 'Expanded plot view';
    if (modalCaption) modalCaption.textContent = caption || '';
    if (imageModal) {
      imageModal.classList.add('open');
      imageModal.setAttribute('aria-hidden', 'false');
    }
  }

  function closeModal() {
    if (imageModal) imageModal.classList.remove('open');
    if (imageModal) imageModal.setAttribute('aria-hidden', 'true');
    if (modalImage) modalImage.src = '';
  }

  function updateElapsed() {
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
    const secs = String(elapsed % 60).padStart(2, '0');
    if (elapsedChip) elapsedChip.textContent = `Elapsed ${mins}:${secs}`;
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting && entry.intersectionRatio > 0.55) {
        activeIndex = slides.indexOf(entry.target);
        renderDots();
        updateTopbar();
      }
    });
  }, { threshold: [0.55, 0.7] });

  slides.forEach((slide) => observer.observe(slide));

  document.querySelectorAll('[data-target]').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = document.getElementById(btn.dataset.target);
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });

  document.querySelectorAll('.stage-card').forEach(card => {
    card.addEventListener('click', () => setArchitectureHighlight(card.dataset.highlight));
  });

  document.querySelectorAll('[data-highlight-target]').forEach(el => {
    el.addEventListener('click', () => setArchitectureHighlight(el.dataset.highlightTarget));
  });

  if (cycleArchitectureBtn) cycleArchitectureBtn.addEventListener('click', cycleArchitecture);

  document.querySelectorAll('.zoomable').forEach(img => {
    img.addEventListener('click', () => {
      openModal(img.src, img.dataset.caption || '', img.alt || 'Expanded image');
    });
  });

  if (imageModal) imageModal.addEventListener('click', (event) => {
    if (event.target === imageModal) closeModal();
  });
  if (modalClose) modalClose.addEventListener('click', closeModal);

  if (fullscreenBtn) fullscreenBtn.addEventListener('click', toggleFullscreen);
  if (presenterBtn) presenterBtn.addEventListener('click', () => togglePresenter());
  if (autoBtn) autoBtn.addEventListener('click', toggleAutoPlay);
  if (startBtn) startBtn.addEventListener('click', () => scrollToSlide(0));

  document.addEventListener('fullscreenchange', () => {
    if (fullscreenBtn) fullscreenBtn.textContent = document.fullscreenElement ? 'Exit Fullscreen' : 'Fullscreen';
  });

  window.addEventListener('keydown', (event) => {
    if (imageModal && imageModal.classList.contains('open') && event.key === 'Escape') {
      closeModal();
      return;
    }
    if (['ArrowDown', 'PageDown', ' '].includes(event.key)) {
      event.preventDefault();
      scrollToSlide(activeIndex + 1);
    }
    if (['ArrowUp', 'PageUp'].includes(event.key)) {
      event.preventDefault();
      scrollToSlide(activeIndex - 1);
    }
    if (event.key === 'Home') { event.preventDefault(); scrollToSlide(0); }
    if (event.key === 'End') { event.preventDefault(); scrollToSlide(slides.length - 1); }
    if (event.key.toLowerCase() === 'f') { event.preventDefault(); toggleFullscreen(); }
    if (event.key.toLowerCase() === 'p') { event.preventDefault(); togglePresenter(); }
    if (event.key.toLowerCase() === 'a') { event.preventDefault(); toggleAutoPlay(); }
  });

  const initialState = getUrlState();
  if (initialState.presenter) togglePresenter(true);
  if (initialState.autoplay) startAutoPlay();

  renderDots();
  updateTopbar();
  updateElapsed();
  if (architectureFigure) setArchitectureHighlight('main-1');
  setInterval(updateElapsed, 1000);
})();
