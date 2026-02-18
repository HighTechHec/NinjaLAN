# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Security Updates

### Recent Security Patches

#### FastAPI ReDoS Vulnerability (2024)
- **Date**: Fixed in current release
- **Severity**: Medium
- **CVE**: CVE-2024-24762 / GHSA-2jv5-9r88-3w3p
- **Description**: Regular Expression Denial of Service in Content-Type header parsing
- **Fix**: Updated FastAPI from 0.104.1 to 0.109.1
- **Impact**: Affects all versions <= 0.109.0
- **Status**: ✅ PATCHED

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

1. **DO NOT** open a public issue
2. Email security concerns to the maintainers or use GitHub Security Advisories
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will:
- Acknowledge receipt within 48 hours
- Provide a detailed response within 7 days
- Work with you to understand and fix the issue
- Credit you in the fix (if desired)

## Security Best Practices

### Deployment

When deploying Second Brain in production:

#### 1. Authentication & Authorization
```yaml
# Enable authentication on all services
NEO4J_AUTH=neo4j/strong-password-here
REDIS_PASSWORD=strong-password-here
```

#### 2. Network Security
```yaml
# Use Docker networks for isolation
networks:
  frontend:  # Public-facing
  backend:   # Internal services only
```

#### 3. TLS/HTTPS
```bash
# Use reverse proxy with TLS
# Example with nginx:
server {
  listen 443 ssl;
  ssl_certificate /path/to/cert.pem;
  ssl_certificate_key /path/to/key.pem;
  
  location / {
    proxy_pass http://localhost:8888;
  }
}
```

#### 4. Environment Variables
```bash
# NEVER commit secrets to git
# Use .env files (gitignored)
# Or use secrets management (Docker secrets, Kubernetes secrets)
```

#### 5. Regular Updates
```bash
# Keep dependencies updated
pip install -r requirements.txt --upgrade

# Update Docker images
docker compose pull
docker compose up -d
```

### Development

#### 1. Dependency Scanning
```bash
# Use pip-audit for vulnerability scanning
pip install pip-audit
pip-audit

# Or use safety
pip install safety
safety check
```

#### 2. Code Scanning
```bash
# Use bandit for security linting
pip install bandit
bandit -r .
```

#### 3. Secret Detection
```bash
# Use gitleaks or truffleHog
# Prevent committing secrets
```

### Infrastructure

#### 1. Docker Security
- Use official base images
- Scan images for vulnerabilities
- Run containers as non-root user
- Limit container resources

#### 2. Service Hardening
- Neo4j: Enable authentication, use bolt+s protocol
- Redis: Set password, disable dangerous commands
- Milvus: Configure access control
- NVIDIA NIM: API key authentication

#### 3. Monitoring
- Enable audit logging
- Monitor for suspicious activity
- Set up alerts for security events
- Regular security audits

## Vulnerability Disclosure Policy

We follow responsible disclosure:

1. Security researcher reports vulnerability privately
2. We confirm and investigate (within 7 days)
3. We develop and test a fix (ASAP, target 30 days)
4. We release patched version
5. We publicly disclose after users have time to update (14 days)
6. We credit the researcher (if desired)

## Dependencies

We maintain a list of critical dependencies:

| Dependency | Purpose | Security Notes |
|------------|---------|----------------|
| FastAPI | Web framework | ✅ Updated to 0.109.1 (ReDoS patched) |
| Pydantic | Data validation | ✅ Updated to 2.5.3 |
| NumPy | Numerical computing | Monitor for updates |
| Neo4j | Knowledge graph | Use authenticated connections |
| Redis | Caching | Set strong password |

## Security Checklist

Before deploying to production:

- [ ] Update all dependencies to latest secure versions
- [ ] Enable authentication on all services (Neo4j, Redis)
- [ ] Configure TLS/HTTPS for API endpoints
- [ ] Use environment variables for secrets (not hardcoded)
- [ ] Set up network isolation (Docker networks)
- [ ] Configure rate limiting on API
- [ ] Enable audit logging
- [ ] Set up monitoring and alerts
- [ ] Review Docker security settings
- [ ] Run security scans (pip-audit, bandit)
- [ ] Test disaster recovery procedures
- [ ] Document security configuration

## Contact

For security concerns:
- GitHub Security Advisories (preferred)
- GitHub Issues (for non-sensitive issues)
- Project maintainers (for critical issues)

---

**Last Updated**: 2024
**Version**: 1.0.0

Stay secure! 🔒
