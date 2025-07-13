import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Search, Trash2, Link2, FileText, Sparkles } from "lucide-react";
import { AnalysisRequest } from "@shared/schema";

interface AnalysisFormProps {
  onAnalyze: (request: AnalysisRequest) => void;
  onClear: () => void;
}

export default function AnalysisForm({ onAnalyze, onClear }: AnalysisFormProps) {
  const [content, setContent] = useState("");
  const [url, setUrl] = useState("");
  const [inputType, setInputType] = useState<"text" | "url">("text");
  const [analysisType, setAnalysisType] = useState<AnalysisRequest["analysisType"]>("comprehensive");
  const [sensitivity, setSensitivity] = useState<AnalysisRequest["sensitivity"]>("standard");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputType === "text" && content.trim()) {
      onAnalyze({
        content: content.trim(),
        analysisType,
        sensitivity,
        inputType: "text",
      });
    } else if (inputType === "url" && url.trim()) {
      onAnalyze({
        content: url.trim(),
        url: url.trim(),
        analysisType,
        sensitivity,
        inputType: "url",
      });
    }
  };

  const handleClear = () => {
    setContent("");
    setUrl("");
    onClear();
  };

  return (
    <Card className="bg-gradient-to-br from-white via-blue-50/30 to-purple-50/30 rounded-3xl shadow-xl border border-gray-200/50 mb-8 backdrop-blur-sm animate-in slide-in-from-bottom-4 duration-500 hover:shadow-2xl transition-all">
      <CardContent className="p-8">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-4">
            <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl p-3 hover:scale-110 transition-transform duration-300">
              <Sparkles className="text-white h-6 w-6" />
            </div>
            <div>
              <h2 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                AI Bias Detection
              </h2>
              <p className="text-sm text-gray-600">Advanced analysis powered by GROQ AI</p>
            </div>
          </div>
          <div className="flex items-center space-x-2 bg-white/60 backdrop-blur-sm rounded-full px-4 py-2">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span className="text-sm font-medium text-gray-700">Ready</span>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-8">
          <Tabs value={inputType} onValueChange={(value) => setInputType(value as "text" | "url")} className="w-full">
            <TabsList className="grid w-full grid-cols-2 bg-white/60 backdrop-blur-sm rounded-xl p-1">
              <TabsTrigger value="text" className="flex items-center space-x-2 rounded-lg transition-all duration-200 hover:scale-105">
                <FileText className="h-4 w-4" />
                <span>📝 Text Content</span>
              </TabsTrigger>
              <TabsTrigger value="url" className="flex items-center space-x-2 rounded-lg transition-all duration-200 hover:scale-105">
                <Link2 className="h-4 w-4" />
                <span>🔗 URL Analysis</span>
              </TabsTrigger>
            </TabsList>
            
            <TabsContent value="text" className="mt-6">
              <div className="space-y-4">
                <Label htmlFor="content" className="text-sm font-semibold text-gray-700">
                  Enter Article or Blog Content for Analysis
                </Label>
                <div className="relative">
                  <Textarea
                    id="content"
                    value={content}
                    onChange={(e) => setContent(e.target.value)}
                    className="min-h-[12rem] resize-none bg-white/80 backdrop-blur-sm border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all duration-200 hover:bg-white/90 hover:scale-[1.02] hover:shadow-lg"
                    placeholder="Paste your article content here for comprehensive bias analysis..."
                    maxLength={5000}
                  />
                  <div className="absolute bottom-3 right-3 text-sm text-gray-400 bg-white/80 backdrop-blur-sm rounded-lg px-2 py-1">
                    {content.length} / 5000 characters
                  </div>
                </div>
              </div>
            </TabsContent>
            
            <TabsContent value="url" className="mt-6">
              <div className="space-y-4">
                <Label htmlFor="url" className="text-sm font-semibold text-gray-700">
                  Enter Article URL for Analysis
                </Label>
                <div className="relative">
                  <Input
                    id="url"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    className="bg-white/80 backdrop-blur-sm border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all duration-200 pl-10 hover:bg-white/90 hover:scale-[1.02] hover:shadow-lg"
                    placeholder="https://example.com/article-to-analyze"
                    type="url"
                  />
                  <Link2 className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                </div>
                <p className="text-sm text-gray-500 bg-blue-50/50 rounded-lg p-3">
                  Enter a URL to automatically extract and analyze article content from blogs, news sites, and other publications.
                </p>
              </div>
            </TabsContent>
          </Tabs>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-3">
              <Label htmlFor="analysisType" className="text-sm font-semibold text-gray-700">
                Analysis Type
              </Label>
              <Select value={analysisType} onValueChange={setAnalysisType}>
                <SelectTrigger className="bg-white/80 backdrop-blur-sm border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all duration-200 hover:bg-white/90 hover:scale-[1.02] hover:shadow-lg">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-white/95 backdrop-blur-sm border-gray-200 rounded-xl">
                  <SelectItem value="comprehensive">🔍 Comprehensive Analysis</SelectItem>
                  <SelectItem value="gender">👥 Gender Bias Only</SelectItem>
                  <SelectItem value="racial">🌍 Racial Bias Only</SelectItem>
                  <SelectItem value="political">🏛️ Political Bias Only</SelectItem>
                  <SelectItem value="cultural">🎭 Cultural Bias Only</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-3">
              <Label htmlFor="sensitivity" className="text-sm font-semibold text-gray-700">
                Sensitivity Level
              </Label>
              <Select value={sensitivity} onValueChange={setSensitivity}>
                <SelectTrigger className="bg-white/80 backdrop-blur-sm border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all duration-200 hover:bg-white/90 hover:scale-[1.02] hover:shadow-lg">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="bg-white/95 backdrop-blur-sm border-gray-200 rounded-xl">
                  <SelectItem value="low">🟢 Low Sensitivity</SelectItem>
                  <SelectItem value="standard">🟡 Standard</SelectItem>
                  <SelectItem value="high">🔴 High Sensitivity</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row gap-4">
            <Button
              type="submit"
              className="flex-1 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-semibold py-3 px-6 rounded-xl transition-all duration-300 transform hover:scale-105 hover:shadow-2xl hover:rotate-1 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
              disabled={inputType === "text" ? !content.trim() : !url.trim()}
            >
              <Search className="h-5 w-5 mr-2 animate-pulse" />
              🚀 Analyze {inputType === "url" ? "URL" : "Content"}
            </Button>
            <Button
              type="button"
              variant="outline"
              className="flex-1 bg-white/80 backdrop-blur-sm border-gray-200 hover:bg-gray-50 font-semibold py-3 px-6 rounded-xl transition-all duration-300 hover:shadow-md hover:scale-105 hover:-rotate-1 active:scale-95"
              onClick={handleClear}
            >
              <Trash2 className="h-5 w-5 mr-2" />
              🗑️ Clear
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
