from jose import jwt, JWTError
from fastapi import HTTPException, status

# Replace with your actual secret and algorithm
SECRET_KEY = "cM7eoGKnSXxHdPaYPB+PF2SQk0wOPoSRIHaZX6GSOmcdNEtK1h9dIK9HT8dDA4YmHuj6PYSNNgOv47Mhi20MJw=="
ALGORITHM = "HS256"  # or RS256 if using public/private key

def decode_jwt(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
