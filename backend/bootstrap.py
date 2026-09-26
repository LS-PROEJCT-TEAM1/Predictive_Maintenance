"""Explicit admin utility; never run at application import/startup."""
import argparse
from firebase_admin import auth
from backend.firebase_service import FirebaseService


def main():
    parser = argparse.ArgumentParser(description='기존 Firebase Auth 계정의 직원 접근 프로필 관리')
    parser.add_argument('--email', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--role', choices=['admin', 'employee'], default='employee')
    parser.add_argument('--inactive', action='store_true')
    args = parser.parse_args()
    service = FirebaseService()
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
