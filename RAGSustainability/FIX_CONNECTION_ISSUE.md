# 🔧 Fix: "App Not Listening on Expected Address"

## 🎯 **Quick Fix - Two Main Issues**

You have **two problems** that I've just fixed:

### **1. Incorrect fly.toml Syntax** ❌➡️✅

**Your fly.toml had wrong syntax:**
```toml
❌ [resources]
   cpu = "shared-cpu-2x"     # Wrong
   memory = 4096             # Wrong format
```

**I fixed it to:**
```toml
✅ [vm]  
   size = "shared-cpu-2x"    # Correct
   memory = "4gb"            # Correct format
```

### **2. Docker Environment Detection** ❌➡️✅

**Enhanced environment detection in app.py:**
- Added multiple detection methods (`/.dockerenv`, `FLY_APP_NAME`)
- Added debugging output to show which mode is detected
- Made Gradio binding more explicit

## 🚀 **Try This Now**

### **Option 1: Quick Re-deploy with Fixes**
```bash
# The syntax is now fixed - redeploy
flyctl deploy

# Watch logs to see the environment detection
flyctl logs -f
```

**Look for this in the logs:**
```
🔍 Environment detection:
   DOCKER_ENV = true
   /.dockerenv exists = True
   FLY_APP_NAME = sustainability-bot-with-3d-plot
   is_docker = True
🐳 Running in Docker mode - binding to 0.0.0.0:7860
```

### **Option 2: Debug Version (Recommended for Testing)**
```bash
# Test with minimal debug app (no vector store loading)
# Update fly.toml to use debug dockerfile
```

Change this line in `fly.toml`:
```toml
[build]
  dockerfile = "Dockerfile.debug"  # Use debug version
```

Then deploy:
```bash
flyctl deploy
flyctl logs -f
```

The debug version should work immediately since it doesn't load the vector store.

## 🔍 **Diagnosing the Issue**

### **Check Current Logs**
```bash
flyctl logs --lines=100
```

**Look for:**
1. **Environment detection output** - Should show `is_docker = True`
2. **Gradio binding message** - Should show "binding to 0.0.0.0:7860"
3. **Any error messages** during startup

### **Test in Container**
```bash
# SSH into your container
flyctl ssh console

# Run environment test
python test_deployment.py

# Check if port is open
curl localhost:7860
```

## ❓ **What You Should See**

### **✅ SUCCESS - Correct Output:**
```
🔍 Environment detection:
   DOCKER_ENV = true
   /.dockerenv exists = True  
   FLY_APP_NAME = sustainability-bot-with-3d-plot
   is_docker = True
🐳 Running in Docker mode - binding to 0.0.0.0:7860
🔧 Gradio will start on all interfaces (0.0.0.0)
Running on public URL: https://0.0.0.0:7860
```

### **❌ PROBLEM - Wrong Output:**
```
🔍 Environment detection:
   DOCKER_ENV = NOT SET
   /.dockerenv exists = False
   FLY_APP_NAME = NOT SET  
   is_docker = False
💻 Running in local mode
```

## 🔧 **If Still Not Working**

### **1. Force Docker Mode**
Edit `app.py` temporarily to force Docker mode:

```python
# Temporary fix - force Docker mode
is_docker = True  # Force this for testing
```

### **2. Check Gradio Version**
```bash
# SSH into container
flyctl ssh console

# Check Gradio version
python -c "import gradio; print(gradio.__version__)"

# Should be 4.0.0 or higher
```

### **3. Test Local Docker Build**
```bash
# Build and test locally
docker build -t rag-test .
docker run -p 7860:7860 -e DOCKER_ENV=true rag-test

# Should bind to 0.0.0.0:7860 and be accessible at localhost:7860
```

## 💡 **Root Cause**

The "not listening on expected address" error happens when:

1. **Wrong fly.toml syntax** → Memory/CPU not allocated properly  
2. **Environment detection fails** → App runs in local mode (127.0.0.1)
3. **Gradio config issues** → Server doesn't bind to 0.0.0.0

**I've fixed #1 and #2. If it still doesn't work, it's likely a Gradio configuration issue.**

## 🎯 **Next Steps**

1. **Try the re-deploy first** (syntax is fixed)
2. **If that fails, use debug version** (isolates the issue)  
3. **Check logs for environment detection output**
4. **SSH into container to test manually if needed**

**The fixes should resolve the binding issue!** 🚀 