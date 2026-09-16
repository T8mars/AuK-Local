using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;

[assembly: AssemblyTitle("AuK Local Launcher")]
[assembly: AssemblyDescription("Portable launcher for the AuK Local speech workstation")]
[assembly: AssemblyCompany("T8star-Aix")]
[assembly: AssemblyProduct("AuK Local")]
[assembly: AssemblyCopyright("Copyright © T8star-Aix 2026")]
[assembly: AssemblyVersion("0.2.3.0")]
[assembly: AssemblyFileVersion("0.2.3.0")]

internal static class AuKLauncher
{
    private static int Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;
        Console.Title = "AuK Local";

        string executable = Assembly.GetExecutingAssembly().Location;
        string packageRoot = Path.GetDirectoryName(executable);
        if (String.IsNullOrEmpty(packageRoot))
        {
            return Fail("无法确定 AuK Local 整合包目录。");
        }

        string python = Path.Combine(packageRoot, "runtime", "python.exe");
        string script = Path.Combine(packageRoot, "scripts", "Start-AuK.cmd");
        string source = Path.Combine(packageRoot, "src", "auk_local");

        if (args.Length == 1 && args[0] == "--print-root")
        {
            Console.WriteLine(packageRoot);
            return 0;
        }

        if (args.Length == 3 && args[0] == "--verify-update-signature")
        {
            return VerifyUpdateSignature(packageRoot, args[1], args[2]);
        }

        if (!File.Exists(python))
        {
            return Fail("整合包不完整，缺少 runtime\\python.exe。\n请完整解压后再启动，不要单独复制 EXE。");
        }
        if (!File.Exists(script))
        {
            return Fail("整合包不完整，缺少 scripts\\Start-AuK.cmd。");
        }
        if (!Directory.Exists(source))
        {
            return Fail("整合包不完整，缺少 src\\auk_local。\n请重新解压完整整合包。");
        }

        if (args.Length == 1 && args[0] == "--verify-package")
        {
            Console.WriteLine("AuK Local launcher: PASS");
            Console.WriteLine("Package root: " + packageRoot);
            return 0;
        }

        try
        {
            ProcessStartInfo info = new ProcessStartInfo();
            info.FileName = Environment.GetEnvironmentVariable("COMSPEC") ?? "cmd.exe";
            // Do not touch ProcessStartInfo.EnvironmentVariables here. On Windows,
            // a parent environment can contain both `Path` and `PATH`; the .NET
            // Framework collection treats those names as the same key and throws
            // before the child process starts. Set the launcher marker inside cmd.
            info.Arguments = "/d /c set AUK_LAUNCHED_BY_EXE=1&& call scripts\\Start-AuK.cmd";
            info.WorkingDirectory = packageRoot;
            info.UseShellExecute = false;
            info.CreateNoWindow = false;

            using (Process process = Process.Start(info))
            {
                if (process == null)
                {
                    return Fail("无法创建 AuK 启动进程。");
                }
                process.WaitForExit();
                if (process.ExitCode == 42)
                {
                    return StartUpdateApplier(packageRoot);
                }
                return process.ExitCode;
            }
        }
        catch (Exception error)
        {
            return Fail("启动 AuK 失败：" + error.Message);
        }
    }

    private static int VerifyUpdateSignature(string packageRoot, string payloadPath, string signatureBase64)
    {
        string publicKeyPath = Path.Combine(packageRoot, "src", "auk_local", "update-public-key.xml");
        if (!File.Exists(publicKeyPath) || !File.Exists(payloadPath))
        {
            return 2;
        }
        try
        {
            byte[] payload = File.ReadAllBytes(payloadPath);
            byte[] signature = Convert.FromBase64String(signatureBase64);
            using (RSACryptoServiceProvider rsa = new RSACryptoServiceProvider())
            {
                rsa.FromXmlString(File.ReadAllText(publicKeyPath, Encoding.UTF8));
                bool valid = rsa.VerifyData(payload, CryptoConfig.MapNameToOID("SHA256"), signature);
                if (valid)
                {
                    Console.WriteLine("AuK update signature: PASS");
                    return 0;
                }
            }
        }
        catch (Exception)
        {
            return 2;
        }
        return 2;
    }

    private static int StartUpdateApplier(string packageRoot)
    {
        string updater = Path.Combine(packageRoot, "scripts", "Apply-AuK-Update.ps1");
        if (!File.Exists(updater))
        {
            return Fail("更新已经准备完成，但缺少 scripts\\Apply-AuK-Update.ps1。");
        }
        try
        {
            ProcessStartInfo info = new ProcessStartInfo();
            info.FileName = "powershell.exe";
            info.Arguments = "-NoProfile -ExecutionPolicy Bypass -File " + Quote(updater)
                + " -PackageRoot " + Quote(packageRoot)
                + " -ParentProcessId " + Process.GetCurrentProcess().Id.ToString();
            info.WorkingDirectory = packageRoot;
            info.UseShellExecute = false;
            info.CreateNoWindow = false;
            Process child = Process.Start(info);
            if (child == null)
            {
                return Fail("无法启动独立更新进程。");
            }
            Console.WriteLine("[AuK] 更新器已接管，正在关闭启动器...");
            return 0;
        }
        catch (Exception error)
        {
            return Fail("无法启动更新器：" + error.Message);
        }
    }

    private static string Quote(string value)
    {
        return "\"" + value.Replace("\"", "\\\"") + "\"";
    }

    private static int Fail(string message)
    {
        Console.Error.WriteLine();
        Console.Error.WriteLine("[AuK 启动器错误] " + message);
        Console.Error.WriteLine();
        if (!Console.IsInputRedirected)
        {
            Console.Error.WriteLine("按任意键关闭此窗口...");
            Console.ReadKey(true);
        }
        return 1;
    }
}
