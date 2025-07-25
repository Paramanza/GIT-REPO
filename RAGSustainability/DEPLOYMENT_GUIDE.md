# 🚀 Deployment Troubleshooting Guide

## 🔍 **Diagnosing Your Issue**

Based on your logs, you're experiencing **Out of Memory (OOM) kills** during vector store loading. Here's how to fix it:

### **Issue Analysis**
```
[info] Out of memory: Killed process 629 (python) total-vm:374436kB, anon-rss:138576kB
[info] INFO Process appears to have been OOM killed!
```

**Root Cause**: Your vector store (~134MB) + Python ML libraries + PCA computation exceeds available memory during startup.

## ✅ **Solution Steps**

### **Step 1: Fix Fly.io Configuration Syntax** 
Your `fly.toml` had incorrect syntax. I've fixed it to:

```toml
[vm]
  size = "shared-cpu-2x"    # Correct syntax
  memory = "4gb"            # String format
```

**Instead of the incorrect:**
```toml
[resources]
  CPU = "shared-cpu-2x"     # Wrong - uppercase CPU
  memory = 4096             # Wrong - should be string
```

### **Step 2: Use High-Memory Configuration** ⭐ **RECOMMENDED**

```bash
# Deploy with 8GB memory to handle vector store loading
cp fly-high-memory.toml fly.toml
flyctl deploy
```

**This configuration:**
- ✅ Uses `dedicated-cpu-1x` with 8GB RAM
- ✅ Extended health checks (5-minute grace period)
- ✅ Optimized for ML workloads
- 💰 Cost: ~$30-50/month (vs $15-20 for shared)

### **Step 3: Alternative - No Health Checks (Debugging)**

If you want to test with your current memory first:

```bash
# Deploy without health checks to isolate the issue
cp fly-no-healthcheck.toml fly.toml
flyctl deploy
```

### **Step 4: Monitor the Deployment**

```bash
# Watch logs in real-time
flyctl logs -f

# Check memory usage after deployment
flyctl status
```

## 🔧 **Quick Fix Commands**

### **Automated Debugging**
```bash
# Run the debugging script (recommended)
chmod +x debug_deployment.sh
./debug_deployment.sh
# Choose option 1: Deploy with high memory
```

### **Manual High-Memory Deploy**
```bash
cp fly-high-memory.toml fly.toml
flyctl deploy
flyctl logs -f
```

## 📊 **Memory Usage Breakdown**

| Component | Memory Usage |
|-----------|--------------|
| Vector Store Loading | ~134MB |
| Python + ML Libraries | ~100-200MB |
| PCA Computation | ~200-400MB (spike) |
| Gradio Interface | ~50-100MB |
| **Total Peak** | **~500-800MB** |

**Current Config**: 4GB (should work, but may hit spikes)  
**Recommended**: 8GB (handles all spikes comfortably)

## 🚨 **What You Should See in Logs**

### **Successful Startup** ✅
```
🚀 Initializing RAG Sustainability Chatbot...
🔍 Environment: Docker
🔑 OpenAI API key loaded: sk-proj-XX...
📚 Loading vector store...
✅ Vector store loaded successfully
📊 Memory usage: 134.2 MB for vectors
🐳 Running in Docker mode - binding to 0.0.0.0:7860
Running on public URL: https://0.0.0.0:7860
```

### **Memory Issues** ❌
```
Out of memory: Killed process 629 (python)
INFO Process appears to have been OOM killed!
```

## 🔄 **Deployment Options**

| Option | Memory | Cost/Month | Use Case |
|--------|--------|------------|----------|
| `fly.toml` | 4GB | $15-20 | Testing (may hit OOM) |
| `fly-high-memory.toml` | 8GB | $30-50 | **Production (recommended)** |
| `fly-no-healthcheck.toml` | 4GB | $15-20 | Debugging only |

## 🛠️ **If High Memory Doesn't Work**

### **1. Check Vector Store Integrity**
```bash
python build_database.py
# This rebuilds the vector store if corrupted
```

### **2. Local Testing**
```bash
# Test Docker build locally
docker build -t rag-test .
docker run -p 7860:7860 -e OPENAI_API_KEY=your_key rag-test
```

### **3. Alternative: Smaller Vector Store**
If budget is a concern, you could:
- Reduce chunk size in `chunker.py`
- Limit document types in knowledge base
- Use lazy loading (requires code changes)

## 📞 **Next Steps**

1. **Immediate Fix**: Run `./debug_deployment.sh` and choose option 1
2. **Monitor**: Watch logs with `flyctl logs -f`
3. **Test**: Access your app with `flyctl open`
4. **Optimize**: Once working, you can experiment with lower memory if desired

## 💡 **Why This Happened**

1. **Vector Store Size**: 134MB of embeddings
2. **Memory Spikes**: PCA computation temporarily doubles memory usage
3. **ML Libraries**: Heavier memory footprint than typical web apps
4. **Config Syntax**: Incorrect Fly.io configuration prevented proper memory allocation

The high-memory configuration should resolve all these issues! 🚀 