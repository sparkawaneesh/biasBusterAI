import { Shield, Globe, Zap, Brain } from "lucide-react";

export default function Header() {
  return (
    <header className="bg-gradient-to-r from-white via-blue-50/30 to-purple-50/30 shadow-lg border-b border-gray-200/50 backdrop-blur-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-center h-20">
          <div className="flex items-center space-x-4">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl blur-sm opacity-60"></div>
              <div className="relative bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl p-3">
                <Shield className="text-white h-7 w-7" />
              </div>
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-gray-900 via-blue-800 to-purple-800 bg-clip-text text-transparent">
                Bias Buster
              </h1>
              <div className="flex items-center space-x-3 mt-1">
                <span className="bg-gradient-to-r from-blue-500 to-purple-600 text-white px-3 py-1 rounded-full text-xs font-semibold flex items-center space-x-1">
                  <Brain className="h-3 w-3" />
                  <span>Python AI</span>
                </span>
                <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-xs font-semibold flex items-center space-x-1">
                  <Zap className="h-3 w-3" />
                  <span>Real-time</span>
                </span>
                <span className="bg-orange-100 text-orange-800 px-3 py-1 rounded-full text-xs font-semibold flex items-center space-x-1">
                  <Globe className="h-3 w-3" />
                  <span>URL Analysis</span>
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
