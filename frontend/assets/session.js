document.addEventListener('click', async event => {
 if (!event.target.closest('#logout-button')) return;
 const csrf=document.cookie.split('; ').find(x=>x.startsWith('manufacturing_csrf='))?.split('=').slice(1).join('=');
 try {const response=await fetch('/auth/logout',{method:'POST',headers:{'X-CSRF-Token':csrf||''}});
 if(response.ok || response.status===401) location.replace('/login');
 else window.alert('로그아웃에 실패했습니다. 새로고침 후 다시 시도하세요.');
 } catch (_) {window.alert('서버 연결을 확인하세요.');}
});
