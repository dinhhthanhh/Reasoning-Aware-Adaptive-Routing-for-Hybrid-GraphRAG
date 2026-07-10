import jwt
import time
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from passlib.context import CryptContext

router = APIRouter()
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

SECRET_KEY = "super-secret-key-for-demo"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 # 1 day

class UserAuth(BaseModel):
    username: str
    password: str

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = time.time() + (ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(authorization: str = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

@router.post("/register")
async def register(user: UserAuth):
    from api.main import get_pipeline
    pipeline = get_pipeline()
    hashed_password = pwd_context.hash(user.password)
    
    success = pipeline.conversation_manager.register_user(user.username, hashed_password)
    if not success:
        raise HTTPException(status_code=400, detail="Username already registered")
        
    return {"message": "User created successfully"}

@router.post("/login")
async def login(user: UserAuth):
    from api.main import get_pipeline
    pipeline = get_pipeline()
    stored_hash = pipeline.conversation_manager.verify_user(user.username)
    
    if not stored_hash or not pwd_context.verify(user.password, stored_hash):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
        
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer", "username": user.username}
