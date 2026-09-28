(() => {
 let pending = false;
 document.addEventListener('click', async event => {
  const button = event.target instanceof Element ? event.target.closest('#logout-button') : null;
  if (!button || pending) return;
  event.preventDefault();
  if (!window.confirm('로그아웃하시겠습니까?\n저장하지 않은 입력 내용은 사라집니다.')) return;
  pending = true;
  const original = button.innerHTML;
  button.disabled = true;
  button.textContent = '종료 중…';
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 15000);
  try {
   const csrf = document.cookie.split('; ').find(x => x.startsWith('manufacturing_csrf='))?.split('=').slice(1).join('=');
   const response = await fetch('/auth/logout', {method:'POST', headers:{'X-CSRF-Token':csrf || ''}, signal:controller.signal});
   if (response.ok || response.status === 401) location.replace('/login');
   else throw new Error('logout failed');
  } catch (_) {
   window.alert('로그아웃을 완료하지 못했습니다. 연결을 확인한 뒤 다시 시도해 주세요.');
  } finally {
   window.clearTimeout(timeout);
   pending = false;
   button.disabled = false;
   button.innerHTML = original;
  }
 });
})();
