"use strict";

// 탭 전환: 매수 목록 ↔ 데이터 읽는 법
(function () {
  const views = {
    list: document.getElementById("view-list"),
    guide: document.getElementById("view-guide"),
  };
  const tabs = document.querySelectorAll("#tabs .tab");

  function show(view) {
    Object.entries(views).forEach(([k, el]) => {
      if (el) el.hidden = k !== view;
    });
    tabs.forEach((t) => t.classList.toggle("is-active", t.dataset.view === view));
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  document.getElementById("tabs").addEventListener("click", (e) => {
    const btn = e.target.closest(".tab");
    if (btn) show(btn.dataset.view);
  });

  // 가이드 안의 "매수 목록으로 돌아가기" 버튼
  document.addEventListener("click", (e) => {
    const jump = e.target.closest(".tab-jump");
    if (jump) show(jump.dataset.view);
  });
})();
