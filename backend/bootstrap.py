"""Explicit admin utility; never run at application import/startup."""
import argparse
from firebase_admin import auth
from backend.firebase_service import FirebaseService


def list_roles(service):
    rows = [(u.email or '-', u.display_name or '-', (u.custom_claims or {}).get('manufacturingRole', '권한 없음'), '비활성' if u.disabled else '활성')
            for u in auth.list_users(app=service.app).iterate_all()]
    for email, name, role, state in sorted(rows, key=lambda r: (r[2] != 'admin', r[0])):
        print(f'{email:<35} {name:<12} {role:<10} {state}')
    print(f'{len(rows)}개 계정')


def main():
    parser = argparse.ArgumentParser(description='기존 Firebase Auth 계정의 직원 접근 프로필 관리')
    parser.add_argument('--list', action='store_true', help='전체 계정의 현재 앱 권한(admin/employee) 조회')
    parser.add_argument('--email')
    parser.add_argument('--name')
    parser.add_argument('--role', choices=['admin', 'employee'], default='employee')
    parser.add_argument('--inactive', action='store_true')
    args = parser.parse_args()
    service = FirebaseService()
    if args.list:
        return list_roles(service)
    if not args.email or not args.name:
        parser.error('--email과 --name을 함께 지정하세요. 조회만 하려면 --list를 사용하세요.')
    user = auth.get_user_by_email(args.email, app=service.app)
    claims = dict(user.custom_claims or {})
    if args.inactive:
        claims.pop('manufacturingRole', None)
    else:
        claims['manufacturingRole'] = args.role
    auth.set_custom_user_claims(user.uid, claims, app=service.app)
    auth.update_user(user.uid, display_name=args.name, app=service.app)
    auth.revoke_refresh_tokens(user.uid, app=service.app)
    print('직원 접근 권한 설정 완료:', args.role, '활성' if not args.inactive else '비활성')


if __name__ == '__main__':
    main()
