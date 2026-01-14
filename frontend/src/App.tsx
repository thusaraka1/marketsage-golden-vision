import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { HashRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import PrivacyPolicy from "./pages/PrivacyPolicy";
import TermsOfService from "./pages/TermsOfService";
import ApiDocs from "./pages/ApiDocs";
import Contact from "./pages/Contact";
import AdminPanel from "./pages/AdminPanel";
import { AuthProvider } from "./context/AuthContext";
import AuthModal from "./components/AuthModal";
import Login from "./pages/Login";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <AuthProvider>
        <Toaster />
        <Sonner />
        <HashRouter>
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/privacy" element={<PrivacyPolicy />} />
            <Route path="/terms" element={<TermsOfService />} />
            <Route path="/api-docs" element={<ApiDocs />} />
            <Route path="/contact" element={<Contact />} />
            <Route path="/login" element={<Login />} />
            <Route path="/admin" element={<AdminPanel />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </HashRouter>
        <AuthModal />
      </AuthProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
