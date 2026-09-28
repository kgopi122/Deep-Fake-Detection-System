import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Mail, Lock, User, ShieldCheck, ArrowRight, Loader2 } from 'lucide-react';
import axios from 'axios';

export default function LoginPage() {
  const [isLogin, setIsLogin] = useState(true);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  
  // Form State
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: ''
  });

  const handleInputChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setErrorMsg("");

    try {
      const endpoint = isLogin ? '/api/auth/login' : '/api/auth/register';
      const payload = isLogin 
        ? { username: formData.username, password: formData.password }
        : formData;

      const apiBase = import.meta.env.VITE_API_URL || '';
      const response = await axios.post(`${apiBase}${endpoint}`, payload);
      
      if (isLogin) {
        // Securely save JWT
        localStorage.setItem("veritrue_token", response.data.token);
        localStorage.setItem("veritrue_user", JSON.stringify(response.data));
        
        // Redirect to dashboard 
        window.location.href = "/";
      } else {
        // Registration successful, flip to login panel
        setIsLogin(true);
        setErrorMsg("Registration successful! Please login.");
      }
    } catch (error) {
      setErrorMsg(error.response?.data?.error || "Connection error. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-neutral-950 flex items-center justify-center p-4 relative overflow-hidden">
      {/* Background Animated Gradient Mesh */}
      <div className="absolute top-[-10%] left-[-10%] w-[40vw] h-[40vw] bg-indigo-600/20 rounded-full blur-[120px] mix-blend-screen animate-pulse" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40vw] h-[40vw] bg-emerald-600/10 rounded-full blur-[120px] mix-blend-screen" />

      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="w-full max-w-md relative z-10"
      >
        <div className="bg-neutral-900/60 backdrop-blur-xl border border-neutral-800 rounded-3xl p-8 shadow-2xl">
          
          {/* Header */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center p-3 bg-neutral-800/50 rounded-2xl mb-4 border border-neutral-700/50">
              <ShieldCheck className="w-8 h-8 text-emerald-400" />
            </div>
            <h1 className="text-3xl font-bold text-white tracking-tight">VeriTrue</h1>
            <p className="text-neutral-400 mt-2 text-sm">
              {isLogin ? "Authenticate to access the detection grid." : "Initialize your security clearance."}
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            
            <AnimatePresence mode="popLayout">
              {!isLogin && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.3 }}
                  className="relative"
                >
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-neutral-500" />
                  <input
                    type="email"
                    name="email"
                    placeholder="Email Address"
                    required={!isLogin}
                    value={formData.email}
                    onChange={handleInputChange}
                    className="w-full bg-neutral-950/50 border border-neutral-800 text-white rounded-xl py-3 pl-10 pr-4 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500 transition-all"
                  />
                </motion.div>
              )}
            </AnimatePresence>

            <div className="relative">
              <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-neutral-500" />
              <input
                type="text"
                name="username"
                placeholder="Username"
                required
                value={formData.username}
                onChange={handleInputChange}
                className="w-full bg-neutral-950/50 border border-neutral-800 text-white rounded-xl py-3 pl-10 pr-4 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500 transition-all"
              />
            </div>

            <div className="relative">
              <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-neutral-500" />
              <input
                type="password"
                name="password"
                placeholder="Password"
                required
                value={formData.password}
                onChange={handleInputChange}
                className="w-full bg-neutral-950/50 border border-neutral-800 text-white rounded-xl py-3 pl-10 pr-4 focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500 transition-all"
              />
            </div>

            {/* Error Message */}
            {errorMsg && (
              <motion.p 
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} 
                className={`text-sm text-center ${errorMsg.includes('successful') ? 'text-emerald-400' : 'text-red-400'}`}
              >
                {errorMsg}
              </motion.p>
            )}

            {/* Submit Button */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              type="submit"
              disabled={loading}
              className="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-medium rounded-xl py-3 mt-4 flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
            >
              {loading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <>
                  {isLogin ? "Initialize Link" : "Register Credentials"}
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </motion.button>
          </form>

          {/* Toggle Login/Register */}
          <div className="mt-6 text-center">
            <button
              onClick={() => { setIsLogin(!isLogin); setErrorMsg(""); }}
              className="text-neutral-400 hover:text-white text-sm transition-colors"
            >
              {isLogin ? "No clearance yet? Request access." : "Already have clearance? Initialize link."}
            </button>
          </div>

        </div>
      </motion.div>
    </div>
  );
}
