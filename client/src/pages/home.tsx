import { useState } from "react";
import Header from "@/components/header";
import AnalysisForm from "@/components/analysis-form";
import LoadingState from "@/components/loading-state";
import ResultsOverview from "@/components/results-overview";
import DetailedReport from "@/components/detailed-report";
import AnalysisHistory from "@/components/analysis-history";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { AnalysisRequest, BiasAnalysis } from "@shared/schema";
import { useToast } from "@/hooks/use-toast";

export default function Home() {
  const [currentAnalysis, setCurrentAnalysis] = useState<BiasAnalysis | null>(null);
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const analyzeMutation = useMutation({
    mutationFn: async (data: AnalysisRequest) => {
      const response = await apiRequest("POST", "/api/analyze", data);
      return response.json();
    },
    onSuccess: (data: BiasAnalysis) => {
      setCurrentAnalysis(data);
      queryClient.invalidateQueries({ queryKey: ["/api/analyses"] });
      toast({
        title: "Analysis Complete",
        description: "Your bias analysis has been completed successfully.",
      });
    },
    onError: (error) => {
      toast({
        title: "Analysis Failed",
        description: "Failed to analyze content. Please try again.",
        variant: "destructive",
      });
      console.error("Analysis error:", error);
    },
  });

  const handleAnalyze = (request: AnalysisRequest) => {
    setCurrentAnalysis(null);
    analyzeMutation.mutate(request);
  };

  const handleClear = () => {
    setCurrentAnalysis(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50/20 to-purple-50/20">
      <Header />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <AnalysisForm onAnalyze={handleAnalyze} onClear={handleClear} />
        
        {analyzeMutation.isPending && <LoadingState />}
        
        {currentAnalysis && (
          <>
            <ResultsOverview analysis={currentAnalysis} />
            <DetailedReport analysis={currentAnalysis} />
          </>
        )}
        
        <AnalysisHistory />
      </main>
    </div>
  );
}
