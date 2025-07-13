import { Card, CardContent } from "@/components/ui/card";
import { Loader2, Brain, Search, Sparkles } from "lucide-react";

export default function LoadingState() {
  return (
    <Card className="bg-gradient-to-br from-white via-blue-50/30 to-purple-50/30 rounded-3xl shadow-xl border border-gray-200/50 mb-8 backdrop-blur-sm">
      <CardContent className="p-8">
        <div className="flex items-center justify-center py-12">
          <div className="flex flex-col items-center">
            <div className="relative mb-8">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full blur-xl opacity-60"></div>
              <div className="relative bg-gradient-to-r from-blue-500 to-purple-600 rounded-full p-6">
                <Brain className="h-12 w-12 text-white" />
              </div>
              <div className="absolute -top-2 -right-2 bg-yellow-400 rounded-full p-2">
                <Sparkles className="h-4 w-4 text-white" />
              </div>
            </div>
            
            <div className="text-center space-y-4">
              <h3 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                AI Analysis in Progress
              </h3>
              <p className="text-gray-600 font-medium">Processing content for bias patterns...</p>
              
              <div className="flex items-center justify-center space-x-4 mt-6">
                <div className="flex items-center space-x-2 bg-white/80 backdrop-blur-sm rounded-full px-4 py-2">
                  <Search className="h-4 w-4 text-blue-500" />
                  <span className="text-sm font-medium text-gray-700">Analyzing</span>
                </div>
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                  <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                  <div className="w-2 h-2 bg-pink-500 rounded-full"></div>
                </div>
              </div>
              
              <div className="mt-8 bg-white/50 backdrop-blur-sm rounded-xl p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">Progress</span>
                  <span className="text-sm font-medium text-gray-700">Processing...</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full" style={{width: '60%'}}></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
