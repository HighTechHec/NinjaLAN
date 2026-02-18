"""
Authentication & Authorization System

Enterprise-grade authentication with:
- JWT token-based authentication
- API key management
- Role-based access control (RBAC)
- Multi-user support with data isolation
- Session management
"""

import secrets
import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Set
from dataclasses import dataclass, field
from enum import Enum
import json

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    print("Warning: PyJWT not available. Install with: pip install PyJWT")


class UserRole(Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    API = "api"


class Permission(Enum):
    """Granular permissions"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    MANAGE_USERS = "manage_users"
    MANAGE_API_KEYS = "manage_api_keys"


# Role-Permission mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: {Permission.READ, Permission.WRITE, Permission.DELETE, 
                     Permission.ADMIN, Permission.MANAGE_USERS, Permission.MANAGE_API_KEYS},
    UserRole.USER: {Permission.READ, Permission.WRITE, Permission.DELETE},
    UserRole.READONLY: {Permission.READ},
    UserRole.API: {Permission.READ, Permission.WRITE}
}


@dataclass
class User:
    """User account"""
    user_id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    created_at: float
    last_login: Optional[float] = None
    is_active: bool = True
    metadata: Dict = field(default_factory=dict)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission"""
        return permission in ROLE_PERMISSIONS.get(self.role, set())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding sensitive data)"""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'role': self.role.value,
            'created_at': self.created_at,
            'last_login': self.last_login,
            'is_active': self.is_active
        }


@dataclass
class APIKey:
    """API key for programmatic access"""
    key_id: str
    key_hash: str
    user_id: str
    name: str
    created_at: float
    last_used: Optional[float] = None
    expires_at: Optional[float] = None
    is_active: bool = True
    permissions: Set[Permission] = field(default_factory=set)
    metadata: Dict = field(default_factory=dict)


@dataclass
class Session:
    """User session"""
    session_id: str
    user_id: str
    created_at: float
    expires_at: float
    last_activity: float
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class AuthManager:
    """
    Authentication and authorization manager
    
    Features:
    - Password hashing with salt
    - JWT token generation and validation
    - API key management
    - Session management
    - RBAC enforcement
    """
    
    def __init__(self, secret_key: Optional[str] = None, token_expiry: int = 3600):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.token_expiry = token_expiry  # seconds
        
        # In-memory storage (in production, use Redis/DB)
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.sessions: Dict[str, Session] = {}
        
        # Username and email indexes
        self.username_index: Dict[str, str] = {}
        self.email_index: Dict[str, str] = {}
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user for initial setup"""
        admin_id = self.create_user(
            username="admin",
            email="admin@secondbrain.local",
            password="admin123",  # Should be changed immediately
            role=UserRole.ADMIN
        )
        print(f"Default admin user created: admin / admin123 (CHANGE THIS!)")
    
    def _hash_password(self, password: str, salt: Optional[str] = None) -> str:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}${pwd_hash.hex()}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, pwd_hex = password_hash.split('$')
            test_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return test_hash.hex() == pwd_hex
        except:
            return False
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.USER
    ) -> str:
        """Create a new user"""
        # Check if username or email already exists
        if username in self.username_index:
            raise ValueError(f"Username '{username}' already exists")
        if email in self.email_index:
            raise ValueError(f"Email '{email}' already exists")
        
        user_id = secrets.token_urlsafe(16)
        password_hash = self._hash_password(password)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            role=role,
            created_at=time.time()
        )
        
        self.users[user_id] = user
        self.username_index[username] = user_id
        self.email_index[email] = user_id
        
        return user_id
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return user_id if successful"""
        user_id = self.username_index.get(username)
        if not user_id:
            return None
        
        user = self.users.get(user_id)
        if not user or not user.is_active:
            return None
        
        if self._verify_password(password, user.password_hash):
            user.last_login = time.time()
            return user_id
        
        return None
    
    def generate_token(self, user_id: str) -> Optional[str]:
        """Generate JWT token for user"""
        if not JWT_AVAILABLE:
            # Fallback to simple token
            return secrets.token_urlsafe(32)
        
        user = self.users.get(user_id)
        if not user:
            return None
        
        payload = {
            'user_id': user_id,
            'username': user.username,
            'role': user.role.value,
            'exp': datetime.utcnow() + timedelta(seconds=self.token_expiry),
            'iat': datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        return token
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token and return payload"""
        if not JWT_AVAILABLE:
            # In fallback mode, we can't verify
            return None
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def create_api_key(
        self,
        user_id: str,
        name: str,
        expires_in: Optional[int] = None,
        permissions: Optional[Set[Permission]] = None
    ) -> str:
        """Create API key for user"""
        user = self.users.get(user_id)
        if not user:
            raise ValueError("User not found")
        
        key = secrets.token_urlsafe(32)
        key_id = secrets.token_urlsafe(16)
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        expires_at = None
        if expires_in:
            expires_at = time.time() + expires_in
        
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            created_at=time.time(),
            expires_at=expires_at,
            permissions=permissions or {Permission.READ, Permission.WRITE}
        )
        
        self.api_keys[key_hash] = api_key
        
        return key  # Return the actual key (only shown once!)
    
    def verify_api_key(self, key: str) -> Optional[APIKey]:
        """Verify API key and return key object"""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        api_key = self.api_keys.get(key_hash)
        
        if not api_key or not api_key.is_active:
            return None
        
        # Check expiration
        if api_key.expires_at and time.time() > api_key.expires_at:
            return None
        
        # Update last used
        api_key.last_used = time.time()
        
        return api_key
    
    def revoke_api_key(self, key_hash: str) -> bool:
        """Revoke an API key"""
        api_key = self.api_keys.get(key_hash)
        if api_key:
            api_key.is_active = False
            return True
        return False
    
    def create_session(
        self,
        user_id: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """Create a new session"""
        session_id = secrets.token_urlsafe(32)
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=time.time(),
            expires_at=time.time() + self.token_expiry,
            last_activity=time.time(),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[session_id] = session
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Session]:
        """Validate session and return if valid"""
        session = self.sessions.get(session_id)
        
        if not session:
            return None
        
        # Check expiration
        if time.time() > session.expires_at:
            del self.sessions[session_id]
            return None
        
        # Update last activity
        session.last_activity = time.time()
        
        return session
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission"""
        user = self.users.get(user_id)
        if not user:
            return False
        
        return user.has_permission(permission)
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def list_users(self) -> List[User]:
        """List all users"""
        return list(self.users.values())
    
    def update_user(
        self,
        user_id: str,
        role: Optional[UserRole] = None,
        is_active: Optional[bool] = None
    ) -> bool:
        """Update user properties"""
        user = self.users.get(user_id)
        if not user:
            return False
        
        if role is not None:
            user.role = role
        if is_active is not None:
            user.is_active = is_active
        
        return True
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user (soft delete - deactivate)"""
        user = self.users.get(user_id)
        if not user:
            return False
        
        user.is_active = False
        return True
    
    def get_user_stats(self, user_id: str) -> Dict:
        """Get statistics for a user"""
        user = self.users.get(user_id)
        if not user:
            return {}
        
        # Count API keys
        api_key_count = sum(1 for key in self.api_keys.values() 
                           if key.user_id == user_id and key.is_active)
        
        # Count active sessions
        session_count = sum(1 for session in self.sessions.values() 
                           if session.user_id == user_id and time.time() < session.expires_at)
        
        return {
            'user': user.to_dict(),
            'api_keys': api_key_count,
            'active_sessions': session_count,
            'permissions': [p.value for p in ROLE_PERMISSIONS.get(user.role, set())]
        }


if __name__ == '__main__':
    # Demo usage
    print("=== Authentication System Demo ===\n")
    
    auth = AuthManager()
    
    # Create users
    print("1. Creating users...")
    user1_id = auth.create_user("alice", "alice@example.com", "password123", UserRole.USER)
    user2_id = auth.create_user("bob", "bob@example.com", "secret456", UserRole.READONLY)
    print(f"   Created Alice: {user1_id}")
    print(f"   Created Bob: {user2_id}")
    
    # Authenticate
    print("\n2. Authentication...")
    auth_id = auth.authenticate("alice", "password123")
    print(f"   Alice authenticated: {auth_id is not None}")
    
    # Generate token
    print("\n3. JWT Token...")
    token = auth.generate_token(user1_id)
    print(f"   Token generated: {token[:20]}...")
    
    # Verify token
    if JWT_AVAILABLE:
        payload = auth.verify_token(token)
        print(f"   Token verified: {payload is not None}")
    
    # Create API key
    print("\n4. API Key...")
    api_key = auth.create_api_key(user1_id, "My API Key", expires_in=86400)
    print(f"   API Key created: {api_key[:20]}...")
    
    # Verify API key
    verified_key = auth.verify_api_key(api_key)
    print(f"   API Key verified: {verified_key is not None}")
    
    # Check permissions
    print("\n5. Permissions...")
    print(f"   Alice can write: {auth.check_permission(user1_id, Permission.WRITE)}")
    print(f"   Bob can write: {auth.check_permission(user2_id, Permission.WRITE)}")
    print(f"   Bob can read: {auth.check_permission(user2_id, Permission.READ)}")
    
    # User stats
    print("\n6. User Statistics...")
    stats = auth.get_user_stats(user1_id)
    print(f"   Alice's stats: {json.dumps(stats, indent=2)}")
    
    print("\n✅ Authentication system demo complete!")
