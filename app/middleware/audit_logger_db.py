from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from datetime import datetime
from app.db.database import SessionLocal
from app.models.audit_log import AuditLog
from app.utils.jwt_utils import decode_jwt

class AuditLoggerDBMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = datetime.utcnow()

        token = request.headers.get("Authorization")
        user_identity = None

        if token and token.startswith("Bearer "):
            try:
                jwt_payload = decode_jwt(token.split(" ")[1])
                user_identity = jwt_payload.get("sub")
            except Exception:
                user_identity = "InvalidToken"

        response = await call_next(request)

        end_time = datetime.utcnow()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        audit_entry = AuditLog(
            timestamp=start_time,
            method=request.method,
            path=str(request.url),
            user_identity=user_identity,
            client_ip=request.client.host,
            user_agent=request.headers.get("User-Agent", ""),
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        # Save to DB
        db = SessionLocal()
        try:
            db.add(audit_entry)
            db.commit()
        except Exception as e:
            db.rollback()
            print("Failed to log audit:", e)
        finally:
            db.close()

        return response
